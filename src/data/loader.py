import os
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None

from src.data.schemas import Tweet, DatasetStats
from src.data.cleaner import TextCleaner


class DataLoader:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def resolve_path(self) -> Path:
        if self.data_path and Path(self.data_path).exists():
            return Path(self.data_path)
        raw_path = Path("data/raw/twcs.csv")
        if raw_path.exists():
            return raw_path
        sample_path = Path("data/raw/sample_twcs.csv")
        if sample_path.exists():
            return sample_path
        raise FileNotFoundError("Neither twcs.csv nor sample_twcs.csv could be found in data/raw/")

    def inspect_dataset(self) -> Tuple[pd.DataFrame, DatasetStats]:
        path = self.resolve_path()
        if duckdb is not None:
            con = duckdb.connect(database=":memory:")
            df = con.execute(f"SELECT * FROM read_csv_auto('{path.as_posix()}')").df()
        else:
            df = pd.read_csv(path.as_posix())

        # Schema detection and missing values
        total_rows = len(df)
        cols = list(df.columns)
        missing = {c: int(df[c].isna().sum()) for c in cols}

        # Handle boolean conversion for inbound
        if "inbound" in df.columns:
            df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "t", "yes"])

        # Determine unique brands and authors
        unique_authors = int(df["author_id"].nunique()) if "author_id" in df.columns else 0
        
        brands = set()
        if "author_id" in df.columns:
            # Non-inbound authors are usually brands
            agent_df = df[~df["inbound"]]
            brands.update(agent_df["author_id"].dropna().unique())

        date_range = None
        if "created_at" in df.columns:
            min_date = str(df["created_at"].min())
            max_date = str(df["created_at"].max())
            date_range = [min_date, max_date]

        # Calculate approximate conversations
        unique_convs = total_rows
        if "in_response_to_tweet_id" in df.columns:
            root_tweets = df[df["in_response_to_tweet_id"].isna()]
            unique_convs = len(root_tweets)

        stats = DatasetStats(
            total_rows=total_rows,
            columns=cols,
            missing_values=missing,
            unique_brands=len(brands),
            unique_authors=unique_authors,
            date_range=date_range,
            unique_conversations=unique_convs
        )
        return df, stats

    def load_tweets(self) -> List[Tweet]:
        df, _ = self.inspect_dataset()
        tweets = []
        for _, row in df.iterrows():
            t_id = str(row.get("tweet_id", "")).strip()
            if not t_id or t_id == "nan":
                continue
            text = TextCleaner.clean_text(str(row.get("text", "")))
            if not text:
                continue

            inbound = str(row.get("inbound", "")).lower() in ["true", "1", "t", "yes"]
            author = str(row.get("author_id", "")).strip()
            
            resp_id = str(row.get("response_tweet_id", "")).strip()
            if resp_id in ["nan", "None", ""]:
                resp_id = None
                
            in_resp_id = str(row.get("in_response_to_tweet_id", "")).strip()
            if in_resp_id in ["nan", "None", ""]:
                in_resp_id = None

            brand = None
            if not inbound:
                brand = author

            tweets.append(Tweet(
                tweet_id=t_id,
                author_id=author,
                inbound=inbound,
                created_at=str(row.get("created_at", "")),
                text=text,
                response_tweet_id=resp_id,
                in_response_to_tweet_id=in_resp_id,
                brand=brand
            ))
        return tweets
