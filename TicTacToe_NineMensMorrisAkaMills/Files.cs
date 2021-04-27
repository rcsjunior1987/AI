using System;
using System.Collections.Generic;
using System.IO;

public static class Files
{
	public static FileInfo File
	{ get; set; }

	public static BoardGame BoardGame
	{ get; set; }

	public static string Path
	{ get; set; }

	private static void SetPath()
	{
		Path = @Environment.CurrentDirectory + "\\" + BoardGame.Name + ".txt";
		File = new FileInfo(Path);
	}
	
	public static void SaveFile()
	{
		SetPath();

		if (System.IO.File.Exists(File.Directory.ToString()))
			File.Delete();

		CreateFile();
	}
	
	private static void SetPiecesOfThePlayers(string txtLines)
	{
		SetPiecesOfThePlayer1(txtLines);
		SetPiecesOfThePlayer2(txtLines);
	}

	private static void SetPiecesOfThePlayer1(string txtLines)
	{
		string lines = txtLines;

		lines = txtLines;

		int initPlayer1Position = txtLines.IndexOf("!");
		int endPlayer1Position = txtLines.IndexOf("?");

		txtLines = txtLines.Substring(initPlayer1Position + 1, (endPlayer1Position - initPlayer1Position) - 1);

		if (txtLines !="")
		{
			string nameOfThePlayer = txtLines.Substring(0, txtLines.IndexOf(","));
			string typeOfThePlayer = txtLines.Substring(txtLines.IndexOf(",")+1
												     , (txtLines.IndexOf("[") - txtLines.IndexOf(",")) -1);

			SetPlayer1FromUpload(typeOfThePlayer, nameOfThePlayer);

			List<Piece> pieces = new List<Piece>();

			while (txtLines != "")
			{
				Piece piece = GetPiecesOfThePlayers(txtLines, out lines);
				piece.FromWhichPlayer = BoardGame.Player1;
				pieces.Add(piece);
				txtLines = lines;
			}

			BoardGame.Player1.Pieces = pieces;
		}
		
	}

	private static void SetPiecesOfThePlayer2(string txtLines)
	{
		string lines = txtLines;

		lines = txtLines;

		int initPlayer2Position = txtLines.IndexOf("+");
		int endPlayer2Position = txtLines.IndexOf("*");

		txtLines = txtLines.Substring(initPlayer2Position + 1, (endPlayer2Position - initPlayer2Position) - 1);

		if (txtLines != "")
		{
			string nameOfThePlayer = txtLines.Substring(0, txtLines.IndexOf(","));
			string typeOfThePlayer = txtLines.Substring(txtLines.IndexOf(",") + 1
													 , (txtLines.IndexOf("[") - txtLines.IndexOf(",")) - 1);

			SetPlayer2FromUpload(typeOfThePlayer, nameOfThePlayer);

			List<Piece> pieces = new List<Piece>();

			while (txtLines != "")
			{
				Piece piece = GetPiecesOfThePlayers(txtLines, out lines);
				piece.FromWhichPlayer = BoardGame.Player2;
				pieces.Add(piece);
				txtLines = lines;
			}

			BoardGame.Player2.Pieces = pieces;
		}
	}

	private static Piece GetPiecesOfThePlayers(string txtLines, out string lines)
	{
		txtLines = txtLines.Substring(txtLines.IndexOf("[")+1
									   , (txtLines.Length - txtLines.IndexOf("["))-1);

		string nameOfThePiece = txtLines.Substring(0, txtLines.IndexOf(","));

		txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1
									   , (txtLines.Length - txtLines.IndexOf(",")) - 1);

		string SymbolOfThePiece = txtLines.Substring(0, txtLines.IndexOf("]"));

		txtLines = txtLines.Substring(txtLines.IndexOf("]") +1
									   , (txtLines.Length - txtLines.IndexOf("]")) - 1);

		Piece piece = new Piece();
		piece.Name = nameOfThePiece;
		piece.Symbol = SymbolOfThePiece;
		lines = txtLines;

		return piece;
	}

	private static void SetCurrentPlayer(string txtLines)
	{
		if (txtLines != "")
		{
			int initCurrentPlayer = txtLines.IndexOf("(");
			int endCurrentPlayer = txtLines.IndexOf(")");

			String currentPlayerName = txtLines.Substring(initCurrentPlayer + 1, (endCurrentPlayer - initCurrentPlayer) - 1);

			if (currentPlayerName == "Player 1")
				BoardGame.CurrentPlayer = BoardGame.Player1;

			if (currentPlayerName == "Player 2")
				BoardGame.CurrentPlayer = BoardGame.Player2;
		}

	}

	private static void SetMills(string txtLines)
	{
		string lines = txtLines;

		int initMillsPosition = txtLines.IndexOf("#");
		int endMillsPosition = txtLines.IndexOf("%");

		txtLines = txtLines.Substring(initMillsPosition + 1, (endMillsPosition - initMillsPosition));

		if(txtLines != "%") {
			BoardGame.Mills.Clear();

			while (txtLines != ""
				&& txtLines != "%")
			{
				int initMillPosition = txtLines.IndexOf("(");
				int endMillPosition = txtLines.IndexOf("%");

				txtLines = txtLines.Substring(initMillPosition + 1, endMillPosition);

				int initPiecePosition = txtLines.IndexOf("[");
				int endPiecePosition = txtLines.IndexOf("]");

				//-------------------------------

				string nameOfThePiece1 = txtLines.Substring(initPiecePosition + 1, txtLines.IndexOf(",") - 1);

				endMillPosition = txtLines.IndexOf("%");

				txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, endMillPosition - 2);

				string SymbolOfThePiece1 = txtLines.Substring(0, txtLines.IndexOf(","));

				endMillPosition = txtLines.IndexOf("%");

				txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, endMillPosition - 5);

				string nameOfThePlayer1 = txtLines.Substring(0, txtLines.IndexOf("]"));

				//-------------------------------

				endMillPosition = txtLines.IndexOf("%"); ;

				txtLines = txtLines.Substring(txtLines.IndexOf("[") + 1, endMillPosition - 9);

				string nameOfThePiece2 = txtLines.Substring(0, txtLines.IndexOf(","));

				endMillPosition = txtLines.IndexOf("%");

				txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, endMillPosition - 1);

				string SymbolOfThePiece2 = txtLines.Substring(0, txtLines.IndexOf(","));

				endMillPosition = txtLines.IndexOf("%");

				txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, endMillPosition - 5);

				string nameOfThePlayer2 = txtLines.Substring(0, txtLines.IndexOf("]"));

				//-------------------------------

				endMillPosition = txtLines.IndexOf("%");

				txtLines = txtLines.Substring(txtLines.IndexOf("[") + 1, endMillPosition - 9);

				string nameOfThePiece3 = txtLines.Substring(0, txtLines.IndexOf(","));

				endMillPosition = txtLines.IndexOf("%");

				txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, endMillPosition - 1);

				string SymbolOfThePiece3 = txtLines.Substring(0, txtLines.IndexOf(","));

				endMillPosition = txtLines.IndexOf("%");

				txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, endMillPosition - 5);

				string nameOfThePlayer3 = txtLines.Substring(0, txtLines.IndexOf("]"));

				//-------------------------------

				endMillPosition = txtLines.IndexOf("%");

				initMillPosition = txtLines.IndexOf("(");

				if (initMillPosition != -1)
					txtLines = txtLines.Substring(txtLines.IndexOf("("), endMillPosition - 9);
				else
					txtLines = "";

				Piece piece1 = new Piece();
				piece1.Name = nameOfThePiece1;
				piece1.Symbol = SymbolOfThePiece1;
				piece1.FromWhichPlayer = (nameOfThePlayer1 == "Player 1") ? BoardGame.Player1 : BoardGame.Player2;

				Piece piece2 = new Piece();
				piece2.Name = nameOfThePiece2;
				piece2.Symbol = SymbolOfThePiece2;
				piece2.FromWhichPlayer = (nameOfThePlayer2 == "Player 1") ? BoardGame.Player1 : BoardGame.Player2;

				Piece piece3 = new Piece();
				piece3.Name = nameOfThePiece3;
				piece3.Symbol = SymbolOfThePiece3;
				piece3.FromWhichPlayer = (nameOfThePlayer2 == "Player 1") ? BoardGame.Player1 : BoardGame.Player2;

				BoardGame.SetMills(piece1, piece2, piece3);
			}
		}

	}

	public static bool SetPiecesOfTheBoard(string txtLines)
	{
		int i = 0;
		int j = BoardGame.BoardItSelf.Length;

		int endBoardPosition = txtLines.IndexOf("!");

		txtLines = txtLines.Substring(0, (endBoardPosition));

		string lines = txtLines;

		while (i <= j-1)
		{
			Coordinates coordinates;
			BoardPosition boardPosition = GetPiecesOfTheBoard(txtLines, out lines, out coordinates);
			BoardGame.BoardItSelf[coordinates.Row, coordinates.Column] = boardPosition;
			txtLines = lines;
			i++;
		}

		return true;
	}

	private static BoardPosition GetPiecesOfTheBoard(string txtLines
												   , out string lines
		                                           , out Coordinates coordinates)
	{
		lines = txtLines;

		BoardPosition boardPosition = new BoardPosition();
		coordinates = new Coordinates();
		if (txtLines != "")
		{
			int initBoardPosition = txtLines.IndexOf("[");
			int endBoardPosition = txtLines.IndexOf("]");

			txtLines = txtLines.Substring(initBoardPosition + 1, (endBoardPosition - initBoardPosition) - 1);

			string row = txtLines.Substring(0, txtLines.IndexOf(","));

			txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, (txtLines.Length - txtLines.IndexOf(",")) - 1);

			string column = txtLines.Substring(0, txtLines.IndexOf(","));

			txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, (txtLines.Length - txtLines.IndexOf(",")) - 1);

			string fromWhichPlayer = txtLines.Substring(0, txtLines.IndexOf(","));

			txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, (txtLines.Length - txtLines.IndexOf(",")) - 1);

			string nameOfThePiece = txtLines.Substring(0, txtLines.IndexOf(","));

			txtLines = txtLines.Substring(txtLines.IndexOf(",") + 1, (txtLines.Length - txtLines.IndexOf(",")) - 1);

			string SymbolOfThePiece = txtLines.Substring(txtLines.IndexOf(",") + 1, (txtLines.Length - (txtLines.IndexOf(",") + 1)));

			lines = lines.Substring(endBoardPosition + 1, (lines.Length - endBoardPosition) - 1);

			boardPosition.Piece.Name = nameOfThePiece;
			boardPosition.Piece.Symbol = SymbolOfThePiece;

			if (fromWhichPlayer == "Player 1")
				boardPosition.Piece.FromWhichPlayer = BoardGame.Player1;

			if (fromWhichPlayer == "Player 2")
				boardPosition.Piece.FromWhichPlayer = BoardGame.Player2;

			coordinates.Row = int.Parse(row);
			coordinates.Column = int.Parse(column);
		}

		return boardPosition;
	}

	private static void SetPlayer1FromUpload(string typeOfThePlayer, string nameOfThePlayer)
	{
		switch (typeOfThePlayer)
		{
			case "HumanPlayer":
				BoardGame.Player1 = new Players().ExecuteCreation(TypeOfPlayer.Human, nameOfThePlayer, null);
				break;
			case "ComputerPlayer":
				BoardGame.Player1 = new Players().ExecuteCreation(TypeOfPlayer.Computer, nameOfThePlayer, null);
				break;
		}
	}

	private static void SetPlayer2FromUpload(string typeOfThePlayer, string nameOfThePlayer)
	{
		switch (typeOfThePlayer)
		{
			case "HumanPlayer":
				BoardGame.Player2 = new Players().ExecuteCreation(TypeOfPlayer.Human, nameOfThePlayer, null);
				break;
			case "ComputerPlayer":
				BoardGame.Player2 = new Players().ExecuteCreation(TypeOfPlayer.Computer, nameOfThePlayer, null);
				break;
		}
	}

	public static bool UploadGame()
	{
		string textRead = UploadFile();

		if (textRead != "")
		{
			SetPiecesOfThePlayers(textRead);
			SetPiecesOfTheBoard(textRead);
			SetCurrentPlayer(textRead);			
			SetMills(textRead);
			return true;
		}

		return false;
	}

	public static string UploadFile()
	{
		SetPath();

		if (System.IO.File.Exists(File.FullName))
			return System.IO.File.ReadAllText(File.FullName);

		return "";
	}

	private static void CreateFile()
	{
		using (StreamWriter sw = File.CreateText())
		{
			string textToBeSaved = "";

			textToBeSaved += BoardGame.GetTextBoardPositions();
			textToBeSaved += BoardGame.GetTextPiecesPlayer1();
			textToBeSaved += BoardGame.GetTextPiecesPlayer2();
			textToBeSaved += BoardGame.GetTextCurrentPlayer();
			textToBeSaved += BoardGame.GetTextMills();

			sw.Write(textToBeSaved);
		}

	}

	

}
