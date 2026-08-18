import requests
import pandas as pd
import time
import os

# STEAM games

GAMES = {
    "Counter-Strike 2": 730,
    "Dota 2": 570,
    "PUBG: BATTLEGROUNDS": 578080,
    "Grand Theft Auto V": 271590,
    "Cyberpunk 2077": 1091500,
    "The Witcher 3": 292030,
    "Stardew Valley": 413150,
    "Terraria": 105600,
    "Portal 2": 620,
    "Left 4 Dead 2": 550
}


# Get reviews for one game

def get_steam_reviews(app_id, max_reviews=1000):

    reviews = []

    cursor = "*"

    url = f"https://store.steampowered.com/appreviews/{app_id}"

    while len(reviews) < max_reviews:

        params = {
            "json": 1,
            "language": "english",
            "filter": "recent",
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": 100,
            "cursor": cursor
        }

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

        except requests.exceptions.RequestException as e:

            print(
                f"Request error for App ID {app_id}: {e}"
            )

            break

        except ValueError:

            print(
                f"Invalid JSON response for App ID {app_id}"
            )

            break

        batch = data.get("reviews", [])

        if not batch:
            print("No more reviews available.")
            break

        for review in batch:

            author = review.get("author", {})

            reviews.append({

                "review_id":
                    review.get("recommendationid"),

                "game_id":
                    app_id,

                "review_text":
                    review.get("review", ""),

                "recommended":
                    review.get("voted_up"),

                "playtime_minutes":
                    author.get(
                        "playtime_forever",
                        0
                    ),

                "helpful_votes":
                    review.get(
                        "votes_up",
                        0
                    ),

                "funny_votes":
                    review.get(
                        "votes_funny",
                        0
                    ),

                "review_date":
                    review.get(
                        "timestamp_created"
                    ),

                "weighted_vote_score":
                    review.get(
                        "weighted_vote_score"
                    ),

                "steam_purchase":
                    review.get(
                        "steam_purchase"
                    ),

                "received_for_free":
                    review.get(
                        "received_for_free"
                    )
            })

            if len(reviews) >= max_reviews:
                break

        cursor = data.get("cursor")

        if not cursor:
            break

        print(
            f"App ID {app_id}: "
            f"{len(reviews)} reviews collected"
        )

        time.sleep(1)

    return pd.DataFrame(reviews)


# Get reviews for multiple games

def scrape_multiple_games(
    games,
    max_reviews_per_game=1000
):

    all_reviews = []

    total_games = len(games)

    for index, (game_name, app_id) in enumerate(
        games.items(),
        start=1
    ):

        print("\n")
        print("=" * 70)
        print(
            f"GAME {index}/{total_games}: "
            f"{game_name}"
        )
        print(
            f"Steam App ID: {app_id}"
        )
        print("=" * 70)

        df = get_steam_reviews(
            app_id=app_id,
            max_reviews=max_reviews_per_game
        )

        if df.empty:

            print(
                f"No reviews found for {game_name}"
            )

            continue

        # Add game name
        df["game_name"] = game_name

        # Put game name near game ID
        columns = [
            "review_id",
            "game_name",
            "game_id",
            "review_text",
            "recommended",
            "playtime_minutes",
            "helpful_votes",
            "funny_votes",
            "review_date",
            "weighted_vote_score",
            "steam_purchase",
            "received_for_free"
        ]

        df = df[
            [
                col for col in columns
                if col in df.columns
            ]
        ]

        all_reviews.append(df)

        print(
            f"\nFinished {game_name}"
        )

        print(
            f"Reviews collected: {len(df)}"
        )

        # Wait before next game
        time.sleep(2)

   
    # Combine all games

    if not all_reviews:

        print(
            "\nNo reviews were collected."
        )

        return pd.DataFrame()

    final_df = pd.concat(
        all_reviews,
        ignore_index=True
    )

    return final_df


# Main

if __name__ == "__main__":

    print("=" * 70)
    print("STEAM MULTI-GAME REVIEW SCRAPER")
    print("=" * 70)

    # Settings

    MAX_REVIEWS_PER_GAME = 1000

    # Scrape

    df = scrape_multiple_games(
        games=GAMES,
        max_reviews_per_game=MAX_REVIEWS_PER_GAME
    )

    # Create output directory

    output_directory = "data/raw"

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # Save CSV

    output_path = (
        f"{output_directory}/"
        "steam_reviews_raw.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    # Results

    print("\n")
    print("=" * 70)
    print("SCRAPING COMPLETED")
    print("=" * 70)

    print(
        f"Total reviews: {len(df)}"
    )

    if not df.empty:

        print(
            f"Total games: "
            f"{df['game_name'].nunique()}"
        )

        print("\nReviews per game:")
        print(
            df["game_name"]
            .value_counts()
        )

    print(
        f"\nDataset saved to:"
        f"\n{output_path}"
    )