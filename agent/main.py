import argparse

from dotenv import load_dotenv


def main():
    load_dotenv()  # load .env file if present
    parser = argparse.ArgumentParser(
        description="AI agent for searching and downloading Raman/SERS spectroscopy datasets"
    )
    parser.add_argument(
        "query",
        nargs="*",
        default=["Raman spectroscopy SERS dataset for machine learning"],
        help="Search query for datasets",
    )
    parser.add_argument(
        "--download-dir",
        default="./downloads",
        help="Directory to save downloaded datasets (default: ./downloads)",
    )

    args = parser.parse_args()
    query = " ".join(args.query)

    from agent.loop import run_agent

    run_agent(query, download_dir=args.download_dir)


if __name__ == "__main__":
    main()
