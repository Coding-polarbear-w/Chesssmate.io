import chess
import chess.pgn
import chess.engine
import streamlit as st
from transformers import pipeline

def analyze_pgn(pgn_file):

    game = chess.pgn.read_game(open(pgn_file))
    board = game.board()
    engine = chess.engine.SimpleEngine.popen_engine("stockfish")

    analysis = ""
    analysis += f"**Event:** {game.headers['Event']}\n"
    analysis += f"**Site:** {game.headers['Site']}\n"
    analysis += f"**Date:** {game.headers['Date']}\n"
    analysis += f"**White:** {game.headers['White']}\n"
    analysis += f"**Black:** {game.headers['Black']}\n"
    analysis += f"**Result:** {game.headers['Result']}\n\n"

    for i, move in enumerate(game.mainline()):
        board.push(move)
        analysis += f"{i+1}. {move.uci()} "

        # Engine evaluation and suggestion:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        evaluation = result.evaluation.score()  # Adjust time limit as needed
        centipawn_eval = evaluation.rel / 100
        analysis += f"({centipawn_eval:.2f} centipawns)"

        if centipawn_eval > 0:
            analysis += " **(good for White)**"
        elif centipawn_eval < 0:
            analysis += " **(good for Black)**"

        # Alternative move suggestion:
        if result.move != move:
            analysis += f", but {result.move.uci()} might have been better"

        analysis += "\n"

    analysis += "\n**Historical similarities:**\n"

    # Use transformers pipeline (potentially fine-tuned on historical chess data)
    model_name = "your_pretrained_model_name"  # Replace with your model name
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    prompt = f"This chess game resembles:\n{analysis}"
    similar_games = pipe(prompt)

    for game in similar_games[:2]:  # Adjust the number of retrieved games
        analysis += f"- {game['generated_text']}\n"

    engine.quit()
    return analysis

def main():
    st.title("Chessmate.io")
    uploaded_file = st.file_uploader("Upload PGN file", type="pgn")

    if uploaded_file is not None:
        analysis = analyze_pgn(uploaded_file.name)
        st.write(analysis)

if __name__ == "__main__":
    main()
