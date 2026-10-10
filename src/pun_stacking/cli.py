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

        if choice == "0":
            run_pipeline()
            continue

        if not choice.isdigit():
            print("Invalid option")
            continue

        index = int(choice)

        if not 1 <= index <= len(STAGES):
            print("Invalid option")
            continue

        try:
            run_stage(STAGES[index - 1].name)
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

    run_parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
    )

    run_parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.command in (None, "menu"):
        show_menu()
        return

    if args.stage == "all":
        if args.split_seed is not None or args.model_seed is not None:
            parser.error("Seed filters cannot be used with the full pipeline")

        run_pipeline()
        return

    if args.stage != "bertimbau":
        if args.split_seed is not None or args.model_seed is not None:
            parser.error("Seed filters are only supported for BERTimbau")

    run_stage(
        args.stage,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
    )


if __name__ == "__main__":
    main()
