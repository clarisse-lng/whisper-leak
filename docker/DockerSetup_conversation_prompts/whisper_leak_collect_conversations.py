#!/usr/bin/env python3
import sys
import os
from core.utils import PrintUtils, OsUtils, ThrowingArgparse, NetworkUtils
from core.chatbot_utils import ChatbotUtils
from core.model import ConversationCollector


def get_self_dir():
    return os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-c', '--chatbot', help='Chatbot to collect from.', default='GPT4oMini')
    parser.add_argument('-s', '--sessions', help='Path to conversations JSON template file.', default='./prompts/conversations/conversations.json')
    parser.add_argument('-o', '--output', help='Output folder for collected data.', default='data/conversations')
    parser.add_argument('-t', '--tlsport', type=int, help='Remote TLS port to sniff.', default=443)
    parser.add_argument('-T', '--temperature', type=float, help='Override temperature.')
    args = parser.parse_args()

    assert args.chatbot, '--chatbot is required'
    assert args.sessions, '--sessions is required'
    assert os.path.isfile(args.sessions), f'Sessions file not found: {args.sessions}'
    assert 0 < args.tlsport <= 0xFFFF, f'Invalid TLS port: {args.tlsport}'

    PrintUtils.end_stage()
    return args


def get_chatbot_class(chatbot_name):
    PrintUtils.start_stage('Loading chatbots')
    chatbots = ChatbotUtils.load_chatbots(os.path.join(get_self_dir(), 'chatbots'))
    assert len(chatbots) > 0, 'Could not load any chatbots'
    PrintUtils.end_stage()

    PrintUtils.start_stage('Initializing chatbot class')
    chatbot_class = chatbots.get(chatbot_name.lower())
    assert chatbot_class is not None, f'Chatbot "{chatbot_name}" does not exist'
    PrintUtils.print_extra(f'Using chatbot *{chatbot_class.__name__}*')
    PrintUtils.end_stage()
    return chatbot_class


def main():
    is_user_cancelled = False
    last_error = None
    aggregated_entries = None

    try:
        PrintUtils.print_logo()
        args = parse_arguments()

        PrintUtils.start_stage('Validating high privileges')
        assert OsUtils.is_high_privileges(), 'User does not run in high privileges'
        PrintUtils.end_stage()

        chatbot_class = get_chatbot_class(args.chatbot)
        out_dir = os.path.join(get_self_dir(), args.output)
        collector = ConversationCollector(args.sessions, out_dir, args.tlsport)
        aggregated_entries = collector.run(chatbot_class, args.temperature)

        dataset_path = collector.get_dataset_path(chatbot_class.__name__, args.temperature)
        PrintUtils.print_extra(
            f'Dataset saved to *{os.path.relpath(dataset_path)}* '
            f'with *{len(aggregated_entries)}* entries.'
        )

    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra('Operation *cancelled* by user — waiting for cleanup')
        is_user_cancelled = True
    except Exception as ex:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex
    finally:
        PrintUtils.start_stage('Running cleanup code')
        NetworkUtils.stop_sniffing_tls(best_effort=True)
        PrintUtils.end_stage()

        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
            sys.exit(1)
        elif is_user_cancelled:
            PrintUtils.print_extra('Operation *cancelled* by user\n')
            sys.exit(1)
        else:
            if aggregated_entries is not None:
                PrintUtils.print_extra(f'Total entries collected: *{len(aggregated_entries)}*')
            PrintUtils.print_extra('Collection finished successfully\n')
            sys.exit(0)


if __name__ == '__main__':
    main()
