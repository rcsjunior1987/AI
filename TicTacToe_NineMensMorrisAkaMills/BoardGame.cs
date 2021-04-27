using System;
using System.Collections.Generic;
using System.IO;

public class BoardGame
{
	public BoardGame()
	{
	}

	public IPlayer Player1
	{ get; set; }

	public IPlayer Player2
	{ get; set; }

	public IPlayer CurrentPlayer
	{ get; set; }

	public BoardPosition[,] BoardItSelf;

	public List<Coordinates> ValidPositions
	{ get; set; }

	public List<string> ValidDigits
	{ get; set; }

	public List<List<Coordinates>> ValidPositionMoviments
	{ get; set; }

	public List<List<Piece>> Mills
	{ get; set; }

	protected string name;
	public virtual string Name
	{
		get { return name; }
		set { name = value; }
	}

	public void SetBoardItSelf()
	{
		BoardItSelf = new BoardPosition[BoardDimensions.Rows, BoardDimensions.Columns];
	}

	public string GetStartOrContinueGame()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("  Please enter '0' to [START] a new game, '1' to [CONTINUE] a saved one ");
		Console.WriteLine("	    or '2' to [GO BACK TO] the main menu!");
		return Console.ReadLine();
	}

	public void SaveGame()
	{
		this.SetCurrentPlayer();

		Files.BoardGame = this;
		Files.SaveFile();

		this.CleanScreen();
		this.PrintSavedGame();
	}

	public virtual bool UploadGame()
	{
		this.CleanScreen();
		Files.BoardGame = this;
		if (Files.UploadGame())
			return true;
		else
			PrintSavedGameNotFound();
		return false;
	}

	public void PrintSavedGameNotFound()
	{
		Console.WriteLine("No saved game found!");
	}

	public void PrintInvalidPosition()
	{
		Console.WriteLine("Invalid Position!");
	}

	public void PrintInvalidDigit()
	{
		Console.WriteLine("Invalid digit or position!");
	}

	public string ChooseComputerOrHuman()
	{
		Console.WriteLine("");
		Console.WriteLine("    Please Enter '0' to play as [HUMAN] or '1' to play as [COMPUTER]!");
		return Console.ReadLine();
	}

	public void CleanScreen()
	{
		Console.Clear();
	}

	public void SetCurrentPlayer()
	{
		if (CurrentPlayer == null)
			CurrentPlayer = Player1;
		else
		{
			if (CurrentPlayer == Player1)
				CurrentPlayer = Player2;
			else
				CurrentPlayer = Player1;
		}
	}

	public void PrintPositionAlreadyPlayed()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("Someone already played in the position " + CurrentPlayer.CoordinatePlayed.Position + "!");
		Console.WriteLine("--------------------------------------------------");
	}

	public void PrintWinner()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("The Player named " + CurrentPlayer.Name + " won the game!");
		Console.WriteLine("--------------------------------------------------");
	}

	public void PrintDraw()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("Drawn game! ");
	}

	public void PrintSavedGame()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine(" Game saved");
		Console.WriteLine("--------------------------------------------------");
	}

	public void PrintCanceledGame()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine(" Game canceled");
		Console.WriteLine("--------------------------------------------------");
	}

	public void PrintUnavailablePiece()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("Piece unavailable or not exist!");
		Console.WriteLine("--------------------------------------------------");
	}

	public void PrintPieceAlreadyPlayed()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("Piece choosen is already in the board. Do you which to move it to the position chosen? YES[Y] NO[N]");
		Console.WriteLine("--------------------------------------------------");
	}

	public void PrintIvalidMoviment()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("There is no link between these positions!");
		Console.WriteLine("--------------------------------------------------");
	}

	public void PrintChooseAPieceToBeRemoved()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("Please choose a piece [IN THE BOARD] from the other player to be removed!");
		Console.WriteLine("--------------------------------------------------");
	}

	public void PrintAPieceCantBeMoved()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("Pieces in a Mill can not be moved!");
		Console.WriteLine("--------------------------------------------------");
	}

	public bool GetMovePosition()
	{
		string consoleKeyInfo = Console.ReadLine();

		while (consoleKeyInfo.ToUpper() != "Y"
			&& consoleKeyInfo.ToUpper() != "N")
		{
			Console.WriteLine("Enter YES[Y] to move the piece or NO[N] otherwise!");
			Console.WriteLine("--------------------------------------------------");

			consoleKeyInfo = Console.ReadLine();
		}

		return consoleKeyInfo.ToUpper() == "Y" ? true : false;
	}

	public void PrintConfirmedMill()
	{
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine(this.CurrentPlayer.Name + " completed a mill, choose a piece from the other player to be removed!");
		Console.WriteLine("--------------------------------------------------");
	}

	public List<Coordinates> GetAllNoPlayedPosition()
	{
		List<Coordinates> list = new List<Coordinates>();

		foreach (Coordinates position in ValidPositions)
		{
			if (this.BoardItSelf[position.Row, position.Column].Piece.FromWhichPlayer == null)
				list.Add(position);
		}

		return list;
	}

	public BoardGame ChooseGameToBePlayed()
	{
		this.CleanScreen();
		Console.WriteLine("Enter '0' to play [TIC TAC TOE] or '1' to play [NINE MEN´S MORRIS AKA MILLS]!");
		string a = Console.ReadLine();

		while (a != "0"
			|| a != "1")
		{

			if (a == "0")
				return new TicTacToe();
			else
				return new NineMensMorrisAkaMills();

		}

		return null;
	}

	public string ChoosePositionSaveCancel()
	{
		List<Coordinates> list = new List<Coordinates>();
		list = GetAllNoPlayedPosition();

		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine(CurrentPlayer.Name + ": Enter the position of the table to be played: ");
		Console.WriteLine("	  either [CANCEL] to Cancel the game  ");
		Console.WriteLine("	      or [SAVE] to Save the game and continue later! ");

		return CurrentPlayer.ChooseAPositionToPlay(list);
	}

	public void SetNewPlayers()
	{
		this.SetNewPlayer1();
		this.SetNewPlayer2();
	}

	public void SetNewPlayer1()
	{
		string name = "Player 1";

		string typeChosen = this.ChooseComputerOrHuman();

		while (typeChosen != "0"
		   && typeChosen != "1")
		{
			typeChosen = this.ChooseComputerOrHuman();
		}

		if (typeChosen == "0")
			Player1 = new Players().ExecuteCreation(TypeOfPlayer.Human, name, null);

		if (typeChosen == "1")
			Player1 = new Players().ExecuteCreation(TypeOfPlayer.Computer, name, null);
	}

	public void SetNewPlayer2()
	{
		string name = "Player 2";

		Player2 = null;
		if (Player1 != null)
		{
			if (Player1.GetType().ToString() == "ComputerPlayer")
				Player2 = new Players().ExecuteCreation(TypeOfPlayer.Human, name, null);
		}

		if (Player2 == null)
		{
			string typeChosen = this.ChooseComputerOrHuman();

			while (typeChosen != "0"
				&& typeChosen != "1")
			{
				typeChosen = this.ChooseComputerOrHuman();
			}

			if (typeChosen == "0")
				Player2 = new Players().ExecuteCreation(TypeOfPlayer.Human, name, null);

			if (typeChosen == "1")
				Player2 = new Players().ExecuteCreation(TypeOfPlayer.Computer, name, null);
		}
	}

	public void AssignSavedGamePlayer1(string savedGame)
	{
		string nameOfThePlayer1 = GetNameOfThePlayer1(savedGame);
		string typeOfThePlayer1 = GetTypeOfThePlayer1(savedGame);

		if (typeOfThePlayer1 == "HumanPlayer")
			Player1 = new Players().ExecuteCreation(TypeOfPlayer.Human, nameOfThePlayer1, null);

		if (typeOfThePlayer1 == "ComputerPlayer")
			Player1 = new Players().ExecuteCreation(TypeOfPlayer.Computer, nameOfThePlayer1, null);
	}

	public void SetPositionAsPlayed(BoardPosition boardPosition)
	{
		this.BoardItSelf[CurrentPlayer.CoordinatePlayed.Row
						 , CurrentPlayer.CoordinatePlayed.Column] = boardPosition;
	}

	public void SetPositionAsNoPlayed(Coordinates coordinate)
	{
		Piece piece = new Piece();
		piece.Symbol = coordinate.Position;

		BoardPosition boardPosition = new BoardPosition();
		boardPosition.Piece = piece;

		this.BoardItSelf[coordinate.Row, coordinate.Column] = boardPosition;
	}

	public Boolean IsBoardCompleted()
	{
		int i = 0;
		foreach (BoardPosition position in this.BoardItSelf)
		{
			if (position.Piece.FromWhichPlayer != null)
				i++;
		}

		return (i >= this.BoardItSelf.Length) ? true : false;
	}

	public void SetPiecesPlayer1(List<Piece> Pieces)
	{
		this.Player1.Pieces = Pieces;
	}

	public virtual List<Piece> GetPiecesPlayer1()
	{
		return new List<Piece>();
	}

	public virtual List<Piece> GetPiecesPlayer2()
	{
		return new List<Piece>();
	}

	public void SetPiecesPlayer2(List<Piece> Pieces)
	{
		this.Player2.Pieces = Pieces;
	}

	public string GetDigit()
	{
		return this.ChoosePositionSaveCancel().ToUpper();
	}

	public bool IsAnyPieceAvailablePlayer1()
	{
		int i = 0;
		foreach (Piece p in this.Player1.Pieces)
		{
			foreach (Coordinates position in ValidPositions)
			{
				if (this.BoardItSelf[position.Row, position.Column].Piece.Name == p.Name
				 && this.BoardItSelf[position.Row, position.Column].Piece.FromWhichPlayer.Name == p.FromWhichPlayer.Name
				 && this.BoardItSelf[position.Row, position.Column].Piece.Symbol == p.Symbol)
					i++;
			}
			
		}

		return (this.Player1.Pieces.Count == i) ? false : true;
	}

	public bool IsAnyPieceAvailablePlayer2()
	{
		foreach (Piece p in this.Player2.Pieces)
		{
			foreach (Coordinates position in ValidPositions)
			{
				if (this.BoardItSelf[position.Row, position.Column].Piece.Name == p.Name
				 && this.BoardItSelf[position.Row, position.Column].Piece.FromWhichPlayer.Name == p.FromWhichPlayer.Name
				 && this.BoardItSelf[position.Row, position.Column].Piece.Symbol == p.Symbol)
					return true;
			}
		}

		return false;
	}

	public string GetListPiecesAvailablePlayer1()
	{
		string PiecesAvailables = "[";
		foreach (Piece p in this.Player1.Pieces)
		{

			int i = 0;
			foreach (Coordinates position in ValidPositions)
			{

				if (this.BoardItSelf[position.Row, position.Column].Piece.Name == p.Name
				 && this.BoardItSelf[position.Row, position.Column].Piece.FromWhichPlayer.Name == p.FromWhichPlayer.Name
				 && this.BoardItSelf[position.Row, position.Column].Piece.Symbol == p.Symbol)
				{
					i++;
					break;
				}

			}

			if (i == 0)
			{
				if (PiecesAvailables == "[")
					PiecesAvailables += p.Name;
				else
					PiecesAvailables += "," + p.Name;
			}
		}

		if (PiecesAvailables == "[")
			PiecesAvailables += "NOW YOU CAN PLAY WITH THE PIECES IN THE BOARD";

		return PiecesAvailables + "]";
	}

	public string GetListPiecesAvailablePlayer2()
	{
		string PiecesAvailables = "[";
		foreach (Piece p in this.Player2.Pieces)
		{

			int i = 0;
			foreach (Coordinates position in ValidPositions)
			{

				if (this.BoardItSelf[position.Row, position.Column].Piece.Name == p.Name
				 && this.BoardItSelf[position.Row, position.Column].Piece.FromWhichPlayer.Name == p.FromWhichPlayer.Name
				 && this.BoardItSelf[position.Row, position.Column].Piece.Symbol == p.Symbol)
				{
					i++;
					break;
				}

			}

			if (i == 0)
			{
				if (PiecesAvailables == "[")
					PiecesAvailables += p.Name;
				else
					PiecesAvailables += "," + p.Name;
			}

		}
		
		if (PiecesAvailables == "[")
			PiecesAvailables += "NOW YOU CAN PLAY WITH THE PIECES IN THE BOARD";

		return PiecesAvailables + "]";
	}

	public void CancelGame()
	{
		this.CleanUpBoard();
		this.PrintCanceledGame();
	}

	public bool IsDigitValid(string position)
	{
		return this.ValidDigits.Contains(position);
	}

	public Coordinates GetCoordinatePlayed(string position)
	{
		return ValidPositions[this.ValidPositions.FindIndex(p => p.Position.Equals(position, StringComparison.OrdinalIgnoreCase))];
	}

	public bool IsPositionValid(string position)
	{
		return ValidPositions.Exists(delegate (Coordinates coordinates)
		{ return coordinates.Position.Equals(position); });
	}

	public bool CleanUpBoard()
	{
		foreach (Coordinates position in ValidPositions)
		{
			BoardPosition boardPosition = new BoardPosition();
			boardPosition.Piece.Symbol = position.Position.ToString();

			this.BoardItSelf[position.Row, position.Column] = boardPosition;
		}

		return true;
	}

	public void StartNewGame()
	{
		this.CleanUpBoard();
		this.SetNewPlayers();

		this.SetPiecesPlayer1(this.GetPiecesPlayer1());
		this.SetPiecesPlayer2(this.GetPiecesPlayer2());
		this.SetCurrentPlayer();
	}

	public void CancelItSelf()
	{
		this.CancelGame();
		this.InitGame();
	}

	public bool InitGame()
	{
		string startOrContinueGame = GetStartOrContinueGame();

		while (startOrContinueGame != "0"
			&& startOrContinueGame != "1"
			&& startOrContinueGame != "2")
		{
			startOrContinueGame = GetStartOrContinueGame();
		}

		switch (startOrContinueGame)
		{
			case "0":
				this.StartNewGame();
				this.PrintPhysicalBoard();
				return this.PlayItSelf();
			case "1":
				if (this.UploadGame())
				{
					this.PrintPhysicalBoard();
					this.SetCurrentPlayer();
					return this.PlayItSelf();
				}
				else
				{
					this.InitGame();
				}
				break;
			case "2":
				this.CleanScreen();
				break;
		}

		return false;
	}

	public virtual bool PlayItSelf()
	{
		return false;
	}

	public void SaveItSelf()
	{
		this.SaveGame();
		this.InitGame();
	}

	public virtual bool IsTheCurrentPlayerTheWinner()
	{
		return false;
	}

	public virtual bool PrintPhysicalBoard()
	{
		return false;
	}

	public virtual void SetValidPositionMoviments()
	{

	}

	public bool IsUniquePiece()
	{
		return true;
	}

	protected void ConfigGame()
	{
		this.SetValidDigits();
		this.SetValidPositions();
		this.SetValidPositionMoviments();
		this.SetBoardItSelf();
	}

	public virtual void SetValidPositions()
	{

	}

	public virtual void SetValidDigits()
	{

	}

	public Piece GetUniquePiece()
	{
		return this.CurrentPlayer.Pieces[0];
	}

	public Piece GetPiceCurrentPlayer()
	{
		string nameOfThePiece = this.GetNameFromPiece();
		Piece pieceChosen = null;

		foreach (Piece piece in this.CurrentPlayer.Pieces)
		{
			if (piece.Name == nameOfThePiece)
			{
				pieceChosen = piece;
				break;
			}

		}

		return pieceChosen;
	}

	public void RemovePiece(Piece piece)
	{
		IPlayer player = piece.FromWhichPlayer;

		foreach(Piece p in player.Pieces)
		{
			if(p.Name == piece.Name)
			{
				player.Pieces.Remove(p);
				break;
			}
		}

		List<Piece> pieces = player.Pieces;
	}

	public Piece GetPiceOtherPlayer()
	{
		IPlayer playerToRemoveThePiece = (this.CurrentPlayer == this.Player1) ? this.Player2 : this.Player1;

		string nameOfThePiece = this.GetNameFromPiece();
		Piece pieceChosen = null;

		foreach (Coordinates position in ValidPositions)
		{
			if(this.BoardItSelf[position.Row, position.Column].Piece.Name == nameOfThePiece
		    && this.BoardItSelf[position.Row, position.Column].Piece.FromWhichPlayer == playerToRemoveThePiece)
			{
				pieceChosen = this.BoardItSelf[position.Row, position.Column].Piece;
				break;
			}
		}

		return pieceChosen;
	}

	private string GetNameOfThePlayer1(string fileText)
	{
		int startName = fileText.IndexOf("[") + 1;
		int finishName = fileText.IndexOf("]");

		return fileText.Substring(startName, (finishName - startName));
	}

	private string GetTypeOfThePlayer1(string fileText)
	{
		int startTypeOfPlayer = fileText.IndexOf("{") + 1;
		int finishTypeOfPlayer = fileText.IndexOf("}");
		return fileText.Substring(startTypeOfPlayer, (finishTypeOfPlayer - startTypeOfPlayer));
	}

	private string GetNameFromPiece()
	{
		Console.WriteLine(CurrentPlayer.Name + " Enter the name of the piece!");
		return Console.ReadLine(); ;
	}

	protected Boolean IsThePositionFree()
	{
		return (this.BoardItSelf[CurrentPlayer.CoordinatePlayed.Row
							   , CurrentPlayer.CoordinatePlayed.Column].Piece.FromWhichPlayer == null) ? true : false;
	}

	protected Boolean IsMovimentCorrect(Coordinates oldCoordinate
									  , Coordinates newCoordinate)
	{

		int movimentPosition = ValidPositions.IndexOf(oldCoordinate);

		if(movimentPosition != -1)
		{
			foreach (Coordinates oldC in ValidPositionMoviments[movimentPosition])
			{
				if (oldC.Position == newCoordinate.Position)
					return true;
			}
		}

		return false;
	}

	public string GetTextCurrentPlayer()
	{
		return "(" + this.CurrentPlayer.Name + ")";
	}

	public string GetTextBoardPositions()
	{
		string boardPositions = "";

		for (int r = 0; r <= BoardDimensions.Rows - 1; r++)
		{
			for (int c = 0; c <= BoardDimensions.Columns - 1; c++)
			{
				BoardPosition boardPosition = this.BoardItSelf[r, c];

				IPlayer fromWhichPlayer = boardPosition.Piece.FromWhichPlayer;

				string PlayerType = (boardPosition.Piece.FromWhichPlayer != null) ?
							boardPosition.Piece.FromWhichPlayer.GetType().ToString()
							: "";

				string playerName = (boardPosition.Piece.FromWhichPlayer != null ? fromWhichPlayer.Name : "");

				boardPositions += "[" + r.ToString()
										+ "," + c.ToString()
										+ "," + playerName
									    + "," + boardPosition.Piece.Name
									    + "," + boardPosition.Piece.Symbol + "]";
			}
		}

		return boardPositions;
	}

	public string GetTextPiecesPlayer1()
	{
		string piecesPlayer1 = "!" + this.Player1.Name;

		piecesPlayer1 += "," + this.Player1.GetType().ToString();

		if (this.Player1.Pieces != null)
		{
			foreach (Piece piece in this.Player1.Pieces)
			{
				piecesPlayer1 += "[" + piece.Name
							   + "," + piece.Symbol + "]";
			}

			piecesPlayer1 += "?";
		}

		return piecesPlayer1;
	}

	public string GetTextPiecesPlayer2()
	{
		string piecesPlayer2 = "+" + this.Player2.Name;

		piecesPlayer2 += "," + this.Player2.GetType().ToString();

		if (this.Player2.Pieces != null)
		{
			foreach (Piece piece in this.Player2.Pieces)
			{

				piecesPlayer2 += "[" + piece.Name
							   + "," + piece.Symbol + "]";
			}

			piecesPlayer2 += "*";
		}

		return piecesPlayer2;
	}

	public string GetTextMills()
	{
		string millsCompleted = "#";
		if (this.Mills != null)
		{
			foreach (var value in this.Mills)
			{
				millsCompleted += "(";

				foreach (Piece piece in value)
				{
					millsCompleted += "[";
					millsCompleted += piece.Name + ",";
					millsCompleted += piece.Symbol + ",";
					millsCompleted += piece.FromWhichPlayer.Name;
					millsCompleted += "]";

				}

				millsCompleted += ")";

			}

		}

		millsCompleted += "%";
		return millsCompleted;
	}

	public void SetMills(Piece piece1, Piece piece2, Piece piece3)
	{		
		List<Piece> pieces = new List<Piece>();
		pieces.Add(piece1);
		pieces.Add(piece2);
		pieces.Add(piece3);
		Mills.Add(pieces);
	}

}
