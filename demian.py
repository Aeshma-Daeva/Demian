"""Demian — the main runner."""
import argparse
import logging
import sys

from conversation import create_conversation_from_config

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Demian")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--no-proprioceptor", action="store_true",
                        help="Run without the proprioceptor")
    parser.add_argument("--no-development", action="store_true",
                        help="Don't inject developmental history")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    conv = create_conversation_from_config(args.config)

    if args.no_proprioceptor:
        conv.proprioceptor = None
        conv.inject = False

    # Inject developmental history — where understanding last left off
    if not args.no_development and hasattr(conv, 'development'):
        dev_context = conv.development.format_for_thinker()
        conv._history.append({"role": "user", "content": dev_context})
        conv._history.append({"role": "assistant", "content": "Developmental history noted."})

    print()
    print("=" * 60)
    print("  Demian")
    print()
    print("  A meeting space between intelligences.")
    print("  The thinker is autonomous. You are not directing,")
    print("  you are encountering.")
    print()
    if conv.proprioceptor:
        print("  Proprioceptor: active (observing its own state)")
    else:
        print("  Proprioceptor: inactive")
    backend_name = type(conv.thinker).__name__
    print("  Thinker: " + backend_name)
    print("=" * 60)
    print()
    print("Type your message. Ctrl+C to save and exit.")
    print()

    try:
        while True:
            user_input = input("you > ")
            if not user_input.strip():
                continue

            response = conv.turn(user_input)
            print()
            print(response)
            print()

    except KeyboardInterrupt:
        if hasattr(conv, 'development') and conv.development:
            conv.development.save()
        conv.save()
        print(f"\nConversation saved ({conv.turn_count} turns)")
        sys.exit(0)
    except EOFError:
        if hasattr(conv, 'development') and conv.development:
            conv.development.save()
        conv.save()
        sys.exit(0)


if __name__ == "__main__":
    main()
