from src.data.database import query_database
from src.data.make_datasets import pull_raw_data

def main():
    pull_raw_data(['contest', 'conversation', 'conversation_feed', 'conversation_like',
                    'conversation_reply', 'follow', 'post', 'post_feed', 'post_like', 'season',
                    'user', 'user_interest', 'video', 'vote'])




if __name__ == "__main__":
    main()