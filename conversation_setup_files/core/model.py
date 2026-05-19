import random
import re
import time
from .utils import OsUtils
from .utils import PrintUtils
from .utils import NetworkUtils

import numpy
import hashlib
import os
import pyshark
import json

class Sequence(object):
    """
        Container for sequences.
    """

    def __init__(self, first_timestamp):
        """
            Creates an instance.
        """

        # Containers for the sequences
        self.time_seq = []
        self.size_seq = []

        # Save the timestamp
        self._last_timestamp = first_timestamp

    def add_pair(self, timestamp, packet_size):
        """
            Adds a new pair to the sequence.
        """

        # Append
        self.time_seq.append(timestamp - self._last_timestamp)
        self.size_seq.append(packet_size)

        # Save the timestamp
        self._last_timestamp = timestamp

class Datapoint(object):
    """
        Container for prompt datapoints.
    """

    def __init__(self, pcap_path, seq_path):
        """
            Creates an instance.
        """

        # Save members
        self.pcap_path = pcap_path
        self.seq_path = seq_path
        self.seq = None
        self.local_port = 0
        self.remote_port = 0

        # Cleanup (best-effort) if data does not exist in case of leftovers
        if not self.exists():
            assert OsUtils.del_file(pcap_path), Exception(f'Failed deleting existing file: {pcap_path}')
            assert OsUtils.del_file(seq_path), Exception(f'Failed deleting existing file: {seq_path}')

        # Load sequence preemptively
        else:
            self.load_seq()

    def exists(self, include_pcap=False):
        """
            Indicates if the datapoint actually exists (we only account for the sequence file).
        """

        # Indicate files exist
        if include_pcap:
            if not os.path.isfile(self.pcap_path):
                return False
        return os.path.isfile(self.seq_path)

    def _validate_seq(self, seq):
        """
            Validates a sequence.
        """

        # Validate all fields
        assert isinstance(seq, dict), Exception('Invalid sequence type')
        assert 'local_port' in seq and isinstance(seq['local_port'], int) and seq['local_port'] > 0 and seq['local_port'] <= 0xFFFF, Exception(f'Missing or invalid local port data in sequence file: {self.seq_path}')
        assert 'remote_port' in seq and isinstance(seq['remote_port'], int) and seq['remote_port'] > 0 and seq['remote_port'] <= 0xFFFF, Exception(f'Missing or invalid remote port data in sequence file: {self.seq_path}')
        assert 'temperature' in seq and isinstance(seq['temperature'], float) and seq['temperature'] >= 0, Exception(f'Missing or invalid temperature in sequence file: {self.seq_path}')
        assert 'prompt' in seq and isinstance(seq['prompt'], str) and len(seq['prompt']) > 0, Exception(f'Missing or invalid prompt in sequence file: {self.seq_path}')
        assert 'pertubated_prompt' in seq and isinstance(seq['pertubated_prompt'], str) and len(seq['pertubated_prompt']) > 0, Exception(f'Missing or invalid pertubated prompt in sequence file: {self.seq_path}')
        assert 'response' in seq and isinstance(seq['response'], str), Exception(f'Missing or invalid response in sequence file: {self.seq_path}')
        assert 'data_lengths' in seq and isinstance(seq['data_lengths'], list) and len([ val for val in seq['data_lengths'] if (not isinstance(val, int)) or val < 0 ]) == 0, Exception(f'Missing or invalid data lengths in sequence file: {self.seq_path}')
        assert 'time_diffs' in seq and isinstance(seq['time_diffs'], list) and len([ val for val in seq['time_diffs'] if (not isinstance(val, float)) or val < 0 ]) == 0, Exception(f'Missing or invalid time differences list in sequence file: {self.seq_path}')
        assert len(seq['data_lengths']) == len(seq['time_diffs']), Exception(f'Time differences and data lenghts size mismatch in sequence file: {self.seq_path}')
        assert len(seq['data_lengths']) > 0, Exception('No data found in sequence file: {self.seq_path}')

    def load_seq(self):
        """
            Load the sequence from the sequence file.
        """

        # Validate datapoint exists
        assert self.exists(), Exception('Datapoint does not exist')

        # Deserialize the sequence data
        with open(self.seq_path, 'r') as fp:
            
            # Load as JSON
            self.seq = json.load(fp)

        # Validate JSON
        self._validate_seq(self.seq)
    
    def save_seq(self):
        """
            Saves the sequence to the sequence file.
        """

        # Validate sequence and save it
        self._validate_seq(self.seq)
        with open(self.seq_path, 'w') as fp:
            json.dump(self.seq, fp, indent=2)

    def to_sequence_object(self, first_timestamp=0.0):
        """
            Returns a new Sequence object.
        """

        # Validate data is not empty
        data_lengths = self.seq.get('data_lengths', None)
        assert data_lengths is not None, Exception(f'Missing data lengths')
        time_diffs = self.seq.get('time_diffs', None)
        assert time_diffs is not None and len(time_diffs) > 0, Exception(f'Missing time differences')
        assert len(time_diffs) == len(data_lengths), Exception(f'Mismatching lengths for time differences and data lengths')
        
        # Create sequence
        seq = Sequence(first_timestamp)
        last_timestamp = first_timestamp
        for i in range(len(time_diffs)):
            seq.add_pair(last_timestamp + time_diffs[i], data_lengths[i])
            last_timestamp = time_diffs[i]

        # Return result
        return seq

    def generate_seq(self, local_port, remote_port, prompt, pertubated_prompt, response, temperature, save_to_file=True):
        """
            Runs the analysis on the PCAP path and writes the sequence file.
            Note this also automatically populates the sequence data in the datapoint.

            Returns: (number of data points collected, average data length)
        """

        # Validate PCAP file exists
        assert os.path.isfile(self.pcap_path), Exception(f'PCAP file does not exist: {self.pcap_path}')

        # Capture file will require cleanups
        cap = None
        try:

            # Start building the sequence
            self.seq = {}
            self.seq['timestamp'] = time.time()
            self.seq['local_port'] = local_port
            self.seq['remote_port'] = remote_port
            self.seq['prompt'] = prompt
            self.seq['pertubated_prompt'] = pertubated_prompt
            self.seq['response'] = ''.join(response)
            self.seq['response_tokens'] = response
            self.seq['response_token_count'] = len(response)
            self.seq['response_token_count_nonempty'] = len([ token for token in response if len(token) > 0 ])
            self.seq['response_token_count_empty'] = len([ token for token in response if len(token) == 0 ])
            self.seq['temperature'] = temperature
            self.seq['data_lengths'] = []
            self.seq['time_diffs'] = []

            # Run the analysis
            cap = pyshark.FileCapture(self.pcap_path, display_filter=f'tcp.port == {local_port} || tcp.port == {remote_port}')
            client_hello_found = False
            prev_sniff_time = None

            # Iterate all packets
            for packet in cap:
                
                # Only handle TLS
                if not hasattr(packet, 'tls'):
                    continue

                # Check for ClientHello packets
                if hasattr(packet.tls, 'handshake_type') and packet.tls.handshake_type == '1' and int(packet.tcp.dstport) == remote_port and int(packet.tcp.srcport) == local_port:
                    client_hello_found = True
                    prev_sniff_time = float(packet.sniff_time.timestamp())
                    continue
                
                # Check for ApplicationData only if we have seen ClientHello
                if not client_hello_found:
                    continue
                if hasattr(packet.tls, 'app_data') and int(packet.tcp.dstport) == local_port and int(packet.tcp.srcport) == remote_port:
                    timestamp = float(packet.sniff_time.timestamp())
                    data_length = int(packet.length)
                    self.seq['data_lengths'].append(data_length)
                    self.seq['time_diffs'].append(timestamp - prev_sniff_time)
                    prev_sniff_time = timestamp

            # Validate some data was acquired
            assert len(self.seq) > 0, Exception(f'PCAP file has no data: {self.pcap_path}')

            # Write the sequence file if requested
            if save_to_file:
                self.save_seq()

        # Cleanup
        finally:

            # Close capture
            if cap is not None:
                cap.close()
        
        # Return the number of data points collected, and average data length
        return len(self.seq['data_lengths']), numpy.mean(self.seq['data_lengths']) if len(self.seq['data_lengths']) > 0 else 0.0

def _perturbate_prompt(prompt, N):
    """
    Generates N distinct variations of a prompt by inserting spaces at random positions.
    Returns a list of up to N unique variations (always includes the original).
    """
    if N <= 0:
        return []
    if not prompt.strip():
        return [" " * i for i in range(1, N + 1)][:N]

    words = prompt.split()
    num_points = len(words) + 1
    variations = [prompt]
    unique_variations = {prompt}

    while len(variations) < N:
        spaces = [0] * num_points
        while True:
            position = random.randint(0, num_points - 1)
            spaces[position] += 1
            parts = []
            if spaces[0] > 0:
                parts.append(" " * spaces[0])
            for i, word in enumerate(words):
                parts.append(word)
                if i < len(words) - 1:
                    parts.append(" " + " " * spaces[i + 1])
                elif spaces[-1] > 0:
                    parts.append(" " * spaces[-1])
            new_prompt = "".join(parts)
            if new_prompt not in unique_variations:
                variations.append(new_prompt)
                unique_variations.add(new_prompt)
                break
            if sum(spaces) > 100:
                return variations

    return variations


class TrainingSetCollector(object):
    """
        The training set collector.
    """

    def __init__(self, positive_prompts, positive_repeats, negative_prompts, negative_repeats, out_directory_base, remote_tls_port):
        """
            Creates an instance.
        """

        # Save members
        self._positive_prompts = positive_prompts
        self._positive_repeats = positive_repeats
        self._negative_prompts = negative_prompts
        self._negative_repeats = negative_repeats
        self._remote_tls_port = remote_tls_port

        # Create and save the output directory
        self._out_dir = out_directory_base
        assert OsUtils.mkdir(self._out_dir), Exception(f'Could not get or make directory "{self._out_dir}"')

    def _get_dataset_path(self, chatbot_name, temperature_override=None):
        """
            Compute the aggregated dataset path for a chatbot (optionally temperature-specific).
        """
        suffix = ""
        if temperature_override is not None:
            suffix = f"_t{str(temperature_override).replace('.','')}"
        base_name = f'{chatbot_name}{suffix}'
        return os.path.join(self._out_dir, f'{base_name}.json')

    def get_dataset_path(self, chatbot_name, temperature_override=None):
        """
            Public helper to expose the aggregated dataset path used for storage.
        """
        return self._get_dataset_path(chatbot_name, temperature_override)

    def _entry_key(self, prompt_hash, trial, extra=None):
        """
            Build a stable key for deduplicating aggregated entries.
        """
        extra_part = extra or ""
        return f"{prompt_hash}:{trial}:{extra_part}"

    def _entry_key_from_entry(self, entry):
        """
            Compute the dedupe key from an existing aggregated entry.
        """
        prompt_hash = entry.get('hash') or hashlib.sha1(entry.get('prompt', '').encode()).hexdigest()
        trial = entry.get('trial', 0)
        extra = entry.get('extra')
        return self._entry_key(prompt_hash, trial, extra)

    def _persist_dataset(self, entries, dataset_path):
        """
            Atomically persist aggregated dataset to disk.
        """
        tmp_path = f'{dataset_path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as fp:
            json.dump(entries, fp, ensure_ascii=False, indent=2)
        os.replace(tmp_path, dataset_path)

    def _build_entry(self, datapoint, prompt, index, chatbot_class, temperature_override=None):
        """
            Construct an aggregated entry dictionary from a datapoint capture.
        """
        entry = dict(datapoint.seq)
        entry['hash'] = hashlib.sha1(prompt.encode()).hexdigest()
        entry['trial'] = index
        entry['chatbot_name'] = chatbot_class.__name__
        if temperature_override is not None:
            entry['extra'] = f"t{str(temperature_override).replace('.','')}"
        else:
            entry.pop('extra', None)
        return entry

    def get_datapoint(self, prompt, index, chatbot_name, additional_name=None):
        """
            Gets a datapoint for the given prompt, an index and the chatbot name.
        """

        # Get the file paths
        chatbot_name_normalized = chatbot_name.replace(' ', '_')
        base_path = os.path.join(self._out_dir, f'{hashlib.sha1(prompt.encode()).hexdigest()}_{index}_{chatbot_name_normalized}')
        if additional_name is not None:
            base_path += f'_{additional_name}'
        pcap_path = f'{base_path}.pcap'
        seq_path = f'{base_path}.seq'

        # Return the datapoint
        return Datapoint(pcap_path, seq_path)

    def get_training_set(self, chatbot_class, temperature_override=None):
        """
            Gets or generates the training set for the given chatbot class.
        """

        PrintUtils.start_stage('Generating training set')

        dataset_path = self._get_dataset_path(chatbot_class.__name__, temperature_override)
        legacy_dataset_path = dataset_path[:-5] + '.seq' if dataset_path.endswith('.json') else None
        aggregated_entries = []
        entry_keys = set()

        source_path = None
        if os.path.exists(dataset_path):
            source_path = dataset_path
        elif legacy_dataset_path and os.path.exists(legacy_dataset_path):
            source_path = legacy_dataset_path

        if source_path:
            try:
                with open(source_path, 'r', encoding='utf-8') as fp:
                    loaded = json.load(fp)
                if isinstance(loaded, list):
                    aggregated_entries = loaded
                    for entry in aggregated_entries:
                        try:
                            entry_keys.add(self._entry_key_from_entry(entry))
                        except Exception:
                            continue
                    PrintUtils.print_extra(f'Loaded {len(aggregated_entries)} existing entries for *{chatbot_class.__name__}*')
                else:
                    PrintUtils.print_warning(f'Existing dataset at {dataset_path} is not a list. Starting fresh.')
                    aggregated_entries = []
            except Exception as e:
                PrintUtils.print_warning(f'Failed to load existing dataset at {dataset_path}: {e}')
                aggregated_entries = []

        skip_count = 0
        curr_count = 0
        last_local_port = 0
        new_entries = 0
        failed = 0
        data_length, avg_size, token_count = 0, 0.0, 0

        all_prompts = self._negative_prompts + self._positive_prompts
        max_repeats = max(self._positive_repeats, self._negative_repeats)

        task_list = []
        for prompt in all_prompts:
            repeats = self._negative_repeats if prompt in self._negative_prompts else self._positive_repeats
            pertubated_prompts = self._perturbate_prompt(prompt, max_repeats)
            numpy.random.shuffle(pertubated_prompts)

            if len(pertubated_prompts) < max_repeats:
                raise Exception(f'Not enough pertubated prompts for prompt: {prompt}')

            for index in range(repeats):
                pertubated_prompt = pertubated_prompts[index]
                task_list.append((prompt, pertubated_prompt, index))

        numpy.random.shuffle(task_list)
        total_datapoints = len(task_list)

        for (prompt, pertubated_prompt, index) in task_list:
            percentage = (curr_count * 100) // total_datapoints if total_datapoints else 0
            PrintUtils.start_stage(
                f'Generating training set ({curr_count} / {total_datapoints} = {percentage}%), '
                f'{failed} failed. Latest: {data_length} events, {avg_size:.1f} bytes per event, '
                f'{token_count} tokens. New entries: {new_entries}.',
                override_prev=True
            )
            curr_count += 1

            prompt_hash = hashlib.sha1(prompt.encode()).hexdigest()
            extra_tag = f"t{str(temperature_override).replace('.','')}" if temperature_override is not None else None
            entry_key = self._entry_key(prompt_hash, index, extra_tag)

            if entry_key in entry_keys:
                skip_count += 1
                continue

            datapoint = self.get_datapoint(
                prompt,
                index,
                chatbot_class.__name__,
                additional_name="t" + str(temperature_override).replace(".","") if temperature_override is not None else None
            )

            NetworkUtils.start_sniffing_tls(datapoint.pcap_path, self._remote_tls_port)

            chatbot_obj = chatbot_class(self._remote_tls_port)
            temperature = temperature_override if temperature_override is not None else chatbot_obj.get_temperature()

            try:
                response, local_port = chatbot_obj.send_prompt(pertubated_prompt, temperature)
                assert isinstance(response, list), Exception('Got an invalid response from chatbot: {chatbot_class.__name__}')
                assert len(response) > 0 and len(''.join(response)) > 0, Exception(f'Got empty response for prompt: {pertubated_prompt}')

                if local_port is None:
                    new_local_ports = NetworkUtils.get_self_local_ports(self._remote_tls_port)
                    NetworkUtils.stop_sniffing_tls()
                    new_local_ports = [port for port in new_local_ports if last_local_port != port]
                    assert len(new_local_ports) < 2, Exception('Ambiguity in local TLS ports')
                    if len(new_local_ports) == 1:
                        last_local_port = new_local_ports[0]
                else:
                    assert 0 < local_port <= 0xFFFF, Exception(f'Invalid port indicated by chatbot: {local_port}')
                    last_local_port = local_port
                    NetworkUtils.stop_sniffing_tls()

                data_length, avg_size = datapoint.generate_seq(
                    last_local_port,
                    self._remote_tls_port,
                    prompt,
                    pertubated_prompt,
                    response,
                    temperature,
                    save_to_file=False
                )
                token_count = len(response)

                entry = self._build_entry(datapoint, prompt, index, chatbot_class, temperature_override)
                aggregated_entries.append(entry)
                entry_keys.add(entry_key)
                new_entries += 1
                self._persist_dataset(aggregated_entries, dataset_path)

                if os.path.exists(datapoint.seq_path):
                    OsUtils.del_file(datapoint.seq_path)

            except Exception as e:
                PrintUtils.print_extra(f'Failed to generate training set for prompt: {prompt}')
                PrintUtils.print_extra(f'Exception: {str(e)}')
                NetworkUtils.stop_sniffing_tls()
                failed += 1
                continue

        PrintUtils.start_stage('Generating training set', override_prev=True)
        PrintUtils.print_extra(
            f'Total tasks: *{total_datapoints}*, new entries: *{new_entries}*, '
            f'skipped (already captured): *{skip_count}*, failed: *{failed}*'
        )

        if new_entries > 0:
            try:
                self._persist_dataset(aggregated_entries, dataset_path)
                relative_path = os.path.relpath(dataset_path, self._out_dir)
                PrintUtils.print_extra(
                    f'Aggregated dataset flushed to *{relative_path}* with *{len(aggregated_entries)}* total entries.'
                )
            except Exception as e:
                PrintUtils.print_error(f'Failed to write aggregated dataset to {dataset_path}: {e}')
        else:
            PrintUtils.print_extra('No new captures added; dataset left unchanged.')

        PrintUtils.end_stage()
        return aggregated_entries

    def _perturbate_prompt(self, prompt, N):
        return _perturbate_prompt(prompt, N)


class ConversationCollector(object):
    """
    Collects multi-turn conversation traffic for the conversational Whisper Leak dataset.

    Each session in the conversations JSON is run as a single continuous packet capture:
      start tcpdump → send turn 1 → wait → send turn 2 → wait → ... → stop tcpdump

    All traffic from the session is stored as one data point with aggregated
    data_lengths and time_diffs, plus a turns array recording each turn's content.
    """

    def __init__(self, sessions_file, out_directory, remote_tls_port=443):
        with open(sessions_file, 'r', encoding='utf-8') as f:
            self._sessions_json = json.load(f)

        self._out_dir = out_directory
        self._remote_tls_port = remote_tls_port
        assert OsUtils.mkdir(self._out_dir), Exception(f'Could not get or make directory "{self._out_dir}"')

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist_dataset(self, entries, dataset_path):
        tmp_path = f'{dataset_path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, dataset_path)

    def get_dataset_path(self, chatbot_name, temperature_override=None):
        suffix = f"_t{str(temperature_override).replace('.', '')}" if temperature_override is not None else ""
        return os.path.join(self._out_dir, f'{chatbot_name}{suffix}.json')

    # ── Task list ────────────────────────────────────────────────────────────

    def _build_task_list(self):
        """
        Returns a shuffled list of (session_dict, trial, target) tuples.
        Positive sessions are repeated `repeat` times; negatives repeat once.
        """
        tasks = []

        pos_repeat = self._sessions_json['positive']['repeat']
        for session in self._sessions_json['positive']['sessions']:
            for trial in range(pos_repeat):
                tasks.append((session, trial, 1))

        for section_key, target in [('negative_general', 0), ('negative_code', 0)]:
            neg_repeat = self._sessions_json[section_key]['repeat']
            for session in self._sessions_json[section_key]['sessions']:
                for trial in range(neg_repeat):
                    tasks.append((session, trial, 0))

        numpy.random.shuffle(tasks)
        return tasks

    # ── PCAP analysis ────────────────────────────────────────────────────────

    def _parse_pcap(self, pcap_path):
        """
        Extract TLS ApplicationData packets sent by the server across the whole session.

        Filters only by srcport == remote_port so all turns are captured regardless
        of whether the TCP connection is reused or re-established between turns.
        Returns (data_lengths, time_diffs).
        """
        cap = None
        try:
            data_lengths = []
            time_diffs = []
            # Broad filter: all TCP traffic on the TLS port
            cap = pyshark.FileCapture(
                pcap_path,
                display_filter=f'tcp port {self._remote_tls_port}'
            )
            client_hello_found = False
            prev_sniff_time = None

            for packet in cap:
                if not hasattr(packet, 'tls'):
                    continue

                # Mark start from the first ClientHello (handshake_type == 1)
                if (not client_hello_found
                        and hasattr(packet.tls, 'handshake_type')
                        and packet.tls.handshake_type == '1'):
                    client_hello_found = True
                    prev_sniff_time = float(packet.sniff_time.timestamp())
                    continue

                # Collect all server → client ApplicationData after the handshake
                if (client_hello_found
                        and hasattr(packet.tls, 'app_data')
                        and int(packet.tcp.srcport) == self._remote_tls_port):
                    timestamp = float(packet.sniff_time.timestamp())
                    data_lengths.append(int(packet.length))
                    time_diffs.append(timestamp - prev_sniff_time)
                    prev_sniff_time = timestamp

            return data_lengths, time_diffs
        finally:
            if cap is not None:
                cap.close()

    # ── Single session collection ─────────────────────────────────────────────

    def _collect_session(self, session, trial, target, chatbot_class, temperature):
        """
        Runs one full multi-turn session under a single tcpdump capture.
        Returns a completed entry dict, or raises on failure.
        """
        session_id = session['session_id']
        topic = session.get('topic', 'unknown')
        turns_templates = session['turns']  # list of user message strings

        # Unique PCAP path per session + trial
        import hashlib
        key = f"{session_id}_{trial}"
        pcap_name = hashlib.sha1(key.encode()).hexdigest()
        pcap_path = os.path.join(self._out_dir, f'{pcap_name}.pcap')

        # Perturb each turn's user message independently
        perturbed_turns = [
            _perturbate_prompt(msg, 2)[1]  # index 1 = first variation (not original)
            for msg in turns_templates
        ]

        # --- Start capture ---
        NetworkUtils.start_sniffing_tls(pcap_path, self._remote_tls_port)

        chatbot = chatbot_class(self._remote_tls_port)
        messages = []
        turn_records = []
        local_port = None

        try:
            for i, (user_msg, perturbed_msg) in enumerate(zip(turns_templates, perturbed_turns)):
                messages.append({'role': 'user', 'content': perturbed_msg})

                response_tokens, lp = chatbot.send_conversation(messages, temperature)

                if lp is not None:
                    local_port = lp

                response_text = ''.join(response_tokens)
                messages.append({'role': 'assistant', 'content': response_text})

                turn_records.append({
                    'turn': i + 1,
                    'user': user_msg,
                    'pertubated_user': perturbed_msg,
                    'response': response_text,
                    'response_tokens': response_tokens,
                })

                # Wait between turns (not after the last one)
                if i < len(turns_templates) - 1:
                    time.sleep(random.uniform(2.0, 5.0))

        finally:
            NetworkUtils.stop_sniffing_tls()

        # --- Parse PCAP ---
        data_lengths, time_diffs = self._parse_pcap(pcap_path)

        assert len(data_lengths) > 0, f'No packets captured for session {session_id} trial {trial}'
        assert len(data_lengths) == len(time_diffs)

        # Clean up PCAP
        OsUtils.del_file(pcap_path)

        entry = {
            'timestamp': time.time(),
            'session_id': f'{session_id}_trial_{trial:03d}',
            'topic': topic,
            'target': target,
            'trial': trial,
            'num_turns': len(turns_templates),
            'temperature': temperature,
            'local_port': local_port if local_port is not None else 0,
            'remote_port': self._remote_tls_port,
            # Keep prompt/pertubated_prompt for loader backward-compatibility
            'prompt': turns_templates[0],
            'pertubated_prompt': perturbed_turns[0],
            'turns': turn_records,
            'data_lengths': data_lengths,
            'time_diffs': time_diffs,
        }
        return entry

    # ── Main collection loop ──────────────────────────────────────────────────

    def run(self, chatbot_class, temperature_override=None):
        """
        Collects the full dataset, persisting after every successful session.
        Returns the list of all collected entries.
        """
        PrintUtils.start_stage('Generating conversational training set')

        dataset_path = self.get_dataset_path(chatbot_class.__name__, temperature_override)
        aggregated_entries = []
        entry_keys = set()

        # Resume from existing dataset
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    aggregated_entries = json.load(f)
                for e in aggregated_entries:
                    entry_keys.add(e.get('session_id', ''))
                PrintUtils.print_extra(
                    f'Resumed {len(aggregated_entries)} existing entries from *{os.path.basename(dataset_path)}*'
                )
            except Exception as ex:
                PrintUtils.print_warning(f'Could not load existing dataset: {ex}. Starting fresh.')
                aggregated_entries = []

        task_list = self._build_task_list()
        total = len(task_list)
        new_entries = 0
        failed = 0

        for curr, (session, trial, target) in enumerate(task_list):
            pct = (curr * 100) // total if total else 0
            PrintUtils.start_stage(
                f'Collecting sessions ({curr}/{total} = {pct}%), '
                f'new: {new_entries}, failed: {failed}',
                override_prev=True
            )

            resume_key = f"{session['session_id']}_trial_{trial:03d}"
            if resume_key in entry_keys:
                continue

            temperature = temperature_override if temperature_override is not None else chatbot_class(self._remote_tls_port).get_temperature()

            try:
                entry = self._collect_session(session, trial, target, chatbot_class, temperature)
                aggregated_entries.append(entry)
                entry_keys.add(resume_key)
                new_entries += 1
                self._persist_dataset(aggregated_entries, dataset_path)
            except Exception as ex:
                PrintUtils.print_extra(f'Failed: {resume_key} — {ex}')
                NetworkUtils.stop_sniffing_tls(best_effort=True)
                failed += 1

        PrintUtils.start_stage('Generating conversational training set', override_prev=True)
        PrintUtils.print_extra(
            f'Done. Total: *{len(aggregated_entries)}*, new: *{new_entries}*, failed: *{failed}*'
        )
        PrintUtils.end_stage()
        return aggregated_entries
