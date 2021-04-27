using System;
using System.Collections.Generic;
using System.IO;

public class TicTacToe : BoardGame, IBoardGame
{
	public TicTacToe()
	{
		BoardDimensions.Rows = 3;
		BoardDimensions.Columns = 3;

		base.Name = "Tic_Tac_Toe";
		base.ConfigGame();
	}

	public override bool PlayItSelf()
	{
		string digit = "";
		
		while (this.IsTheCurrentPlayerTheWinner() == false
			&& base.IsBoardCompleted() == false)
		{
			digit = base.GetDigit();

			if (digit != "CANCEL"
		 	 && digit != "SAVE")
			{
				string position = digit;

				if (base.IsPositionValid(digit))
				{
					this.CurrentPlayer.CoordinatePlayed = base.GetCoordinatePlayed(position);

					if (base.IsThePositionFree())
					{
						BoardPosition boardPosition = new BoardPosition();
						boardPosition.Piece = base.GetUniquePiece();

						this.SetPositionAsPlayed(boardPosition);
						this.PrintPhysicalBoard();

						base.SetCurrentPlayer();
					}
					else {
						base.CleanScreen();
						this.PrintPhysicalBoard();
						base.PrintPositionAlreadyPlayed();
						this.PlayItSelf();
					}
				}
			}
			else
			{
				Console.Clear();

				if (digit == "CANCEL")
					base.CancelItSelf();

				if (digit == "SAVE")
					base.SaveItSelf();

				return false;
			}

		}

		if (digit != "CANCEL"
	  	 && digit != "SAVE")
		{
			if (CurrentPlayer != null)
			{
				if (this.IsTheCurrentPlayerTheWinner() == true)
				{
					base.PrintWinner();
					Console.ReadKey();
				}
				else
				{
					base.PrintDraw();
					Console.ReadKey();
				}
			}
		}

		return true;
	}

	public override void SetValidPositionMoviments()
	{
	}

	public override bool IsTheCurrentPlayerTheWinner()
	{
		if (IsSequence1True()
		 || IsSequence2True()
		 || IsSequence3True()
		 || IsSequence4True()
		 || IsSequence5True()
		 || IsSequence6True()
		 || IsSequence7True()
		 || IsSequence8True()
		 || IsSequence9True())
			return true;
		else
			return false;
	}

	public override void SetValidDigits()
	{
		base.ValidDigits = new List<string>();

		base.ValidDigits.Add("CANCEL");
		base.ValidDigits.Add("SAVE");
		base.ValidDigits.Add("1");
		base.ValidDigits.Add("2");
		base.ValidDigits.Add("3");
		base.ValidDigits.Add("4");
		base.ValidDigits.Add("5");
		base.ValidDigits.Add("6");
		base.ValidDigits.Add("7");
		base.ValidDigits.Add("8");
		base.ValidDigits.Add("9");
	}

	public override void SetValidPositions()
	{
		base.ValidPositions = new List<Coordinates>();

		base.ValidPositions.Add(new Coordinates() { Position = "1", Row = 0, Column = 0 });
		base.ValidPositions.Add(new Coordinates() { Position = "2", Row = 0, Column = 1 });
		base.ValidPositions.Add(new Coordinates() { Position = "3", Row = 0, Column = 2 });
		base.ValidPositions.Add(new Coordinates() { Position = "4", Row = 1, Column = 0 });
		base.ValidPositions.Add(new Coordinates() { Position = "5", Row = 1, Column = 1 });
		base.ValidPositions.Add(new Coordinates() { Position = "6", Row = 1, Column = 2 });
		base.ValidPositions.Add(new Coordinates() { Position = "7", Row = 2, Column = 0 });
		base.ValidPositions.Add(new Coordinates() { Position = "8", Row = 2, Column = 1 });
		base.ValidPositions.Add(new Coordinates() { Position = "9", Row = 2, Column = 2 });
	}

	public override bool PrintPhysicalBoard()
	{
		Console.Clear();

		foreach (Coordinates position in ValidPositions)
		{
			if ((position.Row == 1 && position.Column == 0)
			 || (position.Row == 2 && position.Column == 0))
				Console.WriteLine("");

			Console.Write("      ");

			BoardPosition boardPosition = base.BoardItSelf[position.Row, position.Column];

			Console.Write(boardPosition.Piece.Symbol);
		}

		if (Player1.GetType().ToString() == "ComputerPlayer")
		{
			Console.WriteLine("");
			Console.WriteLine("");
			Console.WriteLine(Player1.Name + " is playing as a computer!");
		}

		if (Player2.GetType().ToString() == "ComputerPlayer")
		{
			Console.WriteLine("");
			Console.WriteLine("");
			Console.WriteLine(Player2.Name + " is playing as a computer!");
		}

		return true;
	}

	public override List<Piece> GetPiecesPlayer1()
	{
		Piece piece = new Piece();
		piece.Name = "1";
		piece.Symbol = "X";
		piece.FromWhichPlayer = base.Player1;

		List<Piece> pieces = new List<Piece>();
		pieces.Add(piece);

		return pieces;
	}

	public override List<Piece> GetPiecesPlayer2()
	{
		Piece piece = new Piece();
		piece.Name = "1";
		piece.Symbol = "O";
		piece.FromWhichPlayer = base.Player2;

		List<Piece> pieces = new List<Piece>();
		pieces.Add(piece);

		return pieces;
	}

	private bool IsSequence1True()
	{ 
		if(base.BoardItSelf[0, 0].Piece.Symbol != null
	    && base.BoardItSelf[1, 1].Piece.Symbol != null
		&& base.BoardItSelf[2, 2].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[0, 0].Piece.Symbol;
			string symbol2 = base.BoardItSelf[1, 1].Piece.Symbol;
			string symbol3 = base.BoardItSelf[2, 2].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
		00 01 02
		10 11 12
		20 21 22
		*/

		return false;
	}

	private bool IsSequence2True()
	{
		if (base.BoardItSelf[0, 2].Piece.Symbol != null
	 	 && base.BoardItSelf[1, 1].Piece.Symbol != null
		 && base.BoardItSelf[2, 0].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[0, 2].Piece.Symbol;
			string symbol2 = base.BoardItSelf[1, 1].Piece.Symbol;
			string symbol3 = base.BoardItSelf[2, 0].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
		00 01 02
		10 11 12
		20 21 22
		*/
		return false;
	}

	private bool IsSequence3True()
	{
		if (base.BoardItSelf[0, 0].Piece.Symbol != null
		 && base.BoardItSelf[0, 1].Piece.Symbol != null
		 && base.BoardItSelf[0, 2].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[0, 0].Piece.Symbol;
			string symbol2 = base.BoardItSelf[0, 1].Piece.Symbol;
			string symbol3 = base.BoardItSelf[0, 2].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
		00 01 02
		10 11 12
		20 21 22
		*/

		return false;
	}

	private bool IsSequence4True()
	{
		if (base.BoardItSelf[1, 0].Piece.Symbol != null
		 && base.BoardItSelf[1, 1].Piece.Symbol != null
		 && base.BoardItSelf[1, 2].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[1, 0].Piece.Symbol;
			string symbol2 = base.BoardItSelf[1, 1].Piece.Symbol;
			string symbol3 = base.BoardItSelf[1, 2].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
		00 01 02
		10 11 12
		20 21 22
		*/

		return false;
	}

	private bool IsSequence5True()
	{
		if (base.BoardItSelf[2, 0].Piece.Symbol != null
		 && base.BoardItSelf[2, 1].Piece.FromWhichPlayer != null
		 && base.BoardItSelf[2, 2].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[2, 0].Piece.Symbol;
			string symbol2 = base.BoardItSelf[1, 0].Piece.Symbol;
			string symbol3 = base.BoardItSelf[2, 0].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
		00 01 02
		10 11 12
		20 21 22
		*/

		return false;
	}

	private bool IsSequence6True()
	{
		if (base.BoardItSelf[0, 0].Piece.Symbol != null
	 	 && base.BoardItSelf[1, 0].Piece.Symbol != null
		 && base.BoardItSelf[2, 0].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[0, 0].Piece.Symbol;
			string symbol2 = base.BoardItSelf[1, 0].Piece.Symbol;
			string symbol3 = base.BoardItSelf[2, 0].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
		00 01 02
		10 11 12
		20 21 22
		*/

		return false;
	}

	private bool IsSequence7True()
	{
		if (base.BoardItSelf[2, 0].Piece.Symbol != null
		 && base.BoardItSelf[2, 1].Piece.Symbol != null
		 && base.BoardItSelf[2, 2].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[2, 0].Piece.Symbol;
			string symbol2 = base.BoardItSelf[2, 1].Piece.Symbol;
			string symbol3 = base.BoardItSelf[2, 2].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
		00 01 02
		10 11 12
		20 21 22
		*/

		return false;
	}

	private bool IsSequence8True()
	{
		if (base.BoardItSelf[0, 1].Piece.Symbol != null
		 && base.BoardItSelf[1, 1].Piece.Symbol != null
		 && base.BoardItSelf[2, 1].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[0, 1].Piece.Symbol;
			string symbol2 = base.BoardItSelf[1, 1].Piece.Symbol;
			string symbol3 = base.BoardItSelf[2, 1].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
			00 01 02
			10 11 12
			20 21 22
			*/

		return false;
	}

	private bool IsSequence9True()
	{
		if (base.BoardItSelf[0, 2].Piece.Symbol != null
		 && base.BoardItSelf[1, 2].Piece.Symbol != null
		 && base.BoardItSelf[2, 2].Piece.Symbol != null)
		{
			string symbol1 = base.BoardItSelf[0, 2].Piece.Symbol;
			string symbol2 = base.BoardItSelf[1, 2].Piece.Symbol;
			string symbol3 = base.BoardItSelf[2, 2].Piece.Symbol;

			return (symbol1 == symbol2 && symbol1 == symbol3 && symbol2 == symbol3) ? true : false;
		}

		/*
			00 01 02
			10 11 12
			20 21 22
			*/

		return false;
	}
}