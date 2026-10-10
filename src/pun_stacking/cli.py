import argparse

from pun_stacking.pipeline import STAGES, run_pipeline, run_stage


def show_menu() -> None:
    while True:
        print()
        print("=" * 50)
        print("PUN STACKING - EXPERIMENTAL PIPELINE")
        print("=" * 50)
        print("0. Run complete pipeline")

        for index, stage in enumerate(STAGES, start=1):
            print(f"{index}. {stage.description}")

        print("q. Exit")
        print()

        choice = input("Select an option: ").strip().lower()

        if choice == "q":
            return

        try:
            if choice == "0":
                run_pipeline()
                continue

            index = int(choice)

            if 1 <= index <= len(STAGES):
                run_stage(STAGES[index - 1].name)
            else:
                print("Invalid option")

        except ValueError:
            print("Invalid option")

        except NotImplementedError as error:
            print(f"Unavailable: {error}")

        except KeyboardInterrupt:
            print("\nOperation interrupted")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Portuguese pun detection experimental pipeline")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("menu")

    run_parser = subparsers.add_parser("run")

    run_parser.add_argument(
        "stage",
        choices=["all", *(stage.name for stage in STAGES)],
    )

    args = parser.parse_args()

    if args.command in (None, "menu"):
        show_menu()
        return

    if args.stage == "all":
        run_pipeline()
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
