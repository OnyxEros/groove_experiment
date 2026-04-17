from IPython.display import Audio, display

def play_index(df, idx):
    path = df.loc[idx, "mp3_path"]
    print("Playing:", path)
    display(Audio(path))
