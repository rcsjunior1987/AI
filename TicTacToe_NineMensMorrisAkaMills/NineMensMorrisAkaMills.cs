using System;
using System.Collections.Generic;

public class NineMensMorrisAkaMills : BoardGame, IBoardGame
{
	public NineMensMorrisAkaMills()
	{
		BoardDimensions.Rows = 4;
		BoardDimensions.Columns = 6;

		base.Name = "Nine_Mens_Morris_Aka_Mills";
		base.ConfigGame();

		base.Mills = new List<List<Piece>>();
	}

	private void SetAllMills()
	{
		/*Piece piece1 = new Piece();
		Piece piece2 = new Piece();
		Piece piece3 = new Piece();

		if (IsSequenceMill1(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill2(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill3(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill4(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill5(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill6(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill7(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill8(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill9(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill10(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill11(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill12(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill13(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill14(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill15(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);

		if (IsSequenceMill16(out piece1, out piece2, out piece3))
			this.SetMills(piece1, piece2, piece3);*/
	}

	public override List<Piece> GetPiecesPlayer1()
	{ 
		Piece piece1 = new Piece();
		piece1.Name = "1";
		piece1.Symbol = "P1{1}";
		piece1.FromWhichPlayer = base.Player1;

		Piece piece2 = new Piece();
		piece2.Name = "2";
		piece2.Symbol = "P1{2}";
		piece2.FromWhichPlayer = base.Player1;

		Piece piece3 = new Piece();
		piece3.Name = "3";
		piece3.Symbol = "P1{3}";
		piece3.FromWhichPlayer = base.Player1;

		Piece piece4 = new Piece();
		piece4.Name = "4";
		piece4.Symbol = "P1{4}";
		piece4.FromWhichPlayer = base.Player1;

		Piece piece5 = new Piece();
		piece5.Name = "5";
		piece5.Symbol = "P1{5}";
		piece5.FromWhichPlayer = base.Player1;

		Piece piece6 = new Piece();
		piece6.Name = "6";
		piece6.Symbol = "P1{6}";
		piece6.FromWhichPlayer = base.Player1;

		Piece piece7 = new Piece();
		piece7.Name = "7";
		piece7.Symbol = "P1{7}";
		piece7.FromWhichPlayer = base.Player1;

		Piece piece8 = new Piece();
		piece8.Name = "8";
		piece8.Symbol = "P1{8}";
		piece8.FromWhichPlayer = base.Player1;

		Piece piece9 = new Piece();
		piece9.Name = "9";
		piece9.Symbol = "P1{9}";
		piece9.FromWhichPlayer = base.Player1;

		List<Piece> pieces = new List<Piece>();
		pieces.Add(piece1);
		pieces.Add(piece2);
		pieces.Add(piece3);
		pieces.Add(piece4);
		pieces.Add(piece5);
		pieces.Add(piece6);
		pieces.Add(piece7);
		pieces.Add(piece8);
		pieces.Add(piece9);

		return pieces;
	}

	public override List<Piece> GetPiecesPlayer2()
	{
		Piece piece1 = new Piece();
		piece1.Name = "1";
		piece1.Symbol = "P2{1}";
		piece1.FromWhichPlayer = base.Player2;

		Piece piece2 = new Piece();
		piece2.Name = "2";
		piece2.Symbol = "P2{2}";
		piece2.FromWhichPlayer = base.Player2;

		Piece piece3 = new Piece();
		piece3.Name = "3";
		piece3.Symbol = "P2{3}";
		piece3.FromWhichPlayer = base.Player2;

		Piece piece4 = new Piece();
		piece4.Name = "4";
		piece4.Symbol = "P2{4}";
		piece4.FromWhichPlayer = base.Player2;

		Piece piece5 = new Piece();
		piece5.Name = "5";
		piece5.Symbol = "P2{5}";
		piece5.FromWhichPlayer = base.Player2;

		Piece piece6 = new Piece();
		piece6.Name = "6";
		piece6.Symbol = "P2{6}";
		piece6.FromWhichPlayer = base.Player2;

		Piece piece7 = new Piece();
		piece7.Name = "7";
		piece7.Symbol = "P2{7}";
		piece7.FromWhichPlayer = base.Player2;

		Piece piece8 = new Piece();
		piece8.Name = "8";
		piece8.Symbol = "P2{8}";
		piece8.FromWhichPlayer = base.Player2;

		Piece piece9 = new Piece();
		piece9.Name = "9";
		piece9.Symbol = "P2{9}";
		piece9.FromWhichPlayer = base.Player2;

		List<Piece> pieces = new List<Piece>();
		pieces.Add(piece1);
		pieces.Add(piece2);
		pieces.Add(piece3);
		pieces.Add(piece4);
		pieces.Add(piece5);
		pieces.Add(piece6);
		pieces.Add(piece7);
		pieces.Add(piece8);
		pieces.Add(piece9);

		return pieces;
	}

	public override bool IsTheCurrentPlayerTheWinner()
	{

		if (base.CurrentPlayer == base.Player1)
		{

			if (this.Player2.Pieces.Count == 0)
				return true;
		}

		if (base.CurrentPlayer == base.Player2)
		{

			if (this.Player1.Pieces.Count == 0)
				return true;
		}

		return false;
	}

	public override bool PlayItSelf()
	{
		string digit = "";

		List<List<Piece>> mills = base.Mills;

		while (this.IsTheCurrentPlayerTheWinner() == false)
		{
			digit = this.GetDigit();

			while (!base.IsDigitValid(digit))
			{
				digit = this.ConfirmInvalidDigit();
			}

			if (digit != "CANCEL"
		 	 && digit != "SAVE")
			{
				string position = digit;
				
				if (base.IsPositionValid(digit))
				{
					this.CurrentPlayer.CoordinatePlayed = base.GetCoordinatePlayed(position);

					if (base.IsThePositionFree())
					{
						Piece piece = base.GetPiceCurrentPlayer();

						while (piece == null)
						{
							piece = ConfirmUnavaliblePiece();
						}

						Coordinates oldCoordinate = this.GetCoordinatesOfAPiece(piece);

						if (IsPieceNoInAnyMill(piece))
						{

							while (oldCoordinate != null)
							{
								this.PrintPhysicalBoard();
								base.PrintPieceAlreadyPlayed();
						
								if (base.GetMovePosition())
								{
									if (IsMovimentCorrect(oldCoordinate
														, this.CurrentPlayer.CoordinatePlayed))
										this.MoveAPieceInTheBoard(piece, oldCoordinate);
									else
										this.ConfirmInvalidMoviment();

									break;

								}
								else
								{
									//this.PrintPhysicalBoard();
									break;
								}

							}

						}
						else
						{
							this.ConfirmPieceCantBeMoved();
							//break;
						}

						if (oldCoordinate == null)
							this.MoveAPieceToTheBoard(piece);
					}
					else
						this.ConfirmPositionAlreadyPlayed();
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
				base.PrintWinner();
				Console.ReadKey();
			}
		}

		return true;
	}

	private void ConfirmPieceCantBeMoved()
	{
		this.PrintPhysicalBoard();
		this.PrintAPieceCantBeMoved();
	}

	private string ConfirmInvalidDigit()
	{
		this.PrintPhysicalBoard();
		base.PrintInvalidDigit();
		return this.ChoosePositionSaveCancel().ToUpper();
	}

	private Piece ConfirmUnavaliblePiece()
	{
		this.PrintPhysicalBoard();
		base.PrintUnavailablePiece();
		return base.GetPiceCurrentPlayer();
	}

	private void ConfirmPositionAlreadyPlayed()
	{
		this.PrintPhysicalBoard();
		base.PrintPositionAlreadyPlayed();
		this.PlayItSelf();
	}

	private void ConfirmInvalidMoviment()
	{
		this.PrintPhysicalBoard();
		base.PrintIvalidMoviment();
	}

	private void MoveAPieceToTheBoard(Piece piece)
	{
		BoardPosition boardPosition = new BoardPosition();
		boardPosition.Piece = piece;

		base.SetPositionAsPlayed(boardPosition);
		this.PrintPhysicalBoard();

		this.ConfirmMill();
		base.SetCurrentPlayer();
	}

	private void MoveAPieceInTheBoard(Piece piece, Coordinates oldCoordinate)
	{
		BoardPosition boardPosition = new BoardPosition();
		boardPosition.Piece = piece;

		base.SetPositionAsPlayed(boardPosition);
		base.SetPositionAsNoPlayed(oldCoordinate);

		this.PrintPhysicalBoard();

		this.ConfirmMill();
		base.SetCurrentPlayer();
	}

	public override bool PrintPhysicalBoard()
	{
		Console.Clear();

		string line1 = this.GetPrintLine1();
		string line2 = this.GetPrintLine2();
		string line3 = this.GetPrintLine3();
		string line4 = this.GetPrintLine4();
		string line5 = this.GetPrintLine5();
		string line6 = this.GetPrintLine6();
		string line7 = this.GetPrintLine7();

		string PiecesAvailable = "\n\n---------------- Pieces availables --------------------";
		PiecesAvailable += "\n             " + base.Player1.Name + base.GetListPiecesAvailablePlayer1();
		PiecesAvailable += "\n             " + base.Player2.Name + base.GetListPiecesAvailablePlayer2();
		PiecesAvailable += "\n-------------------------------------------------------";

		Console.Write(line1
					+ line2
					+ line3
					+ line4
					+ line5
					+ line6
					+ line7);

		Console.WriteLine(PiecesAvailable);

		//0,0 0,1 0,3 0,4 0,5 0,6
		//1,0 1,1 1,3 1,4 1,5 1,6
		//2,0 2,1 2,3 2,4 2,5 2,6
		//3,0 3,1 3,3 3,4 3,5 3,6

		return true;
	}

	public override void SetValidDigits()
	{
		base.ValidDigits = new List<string>();

		base.ValidDigits.Add("CANCEL");
		base.ValidDigits.Add("SAVE");
		base.ValidDigits.Add("A");
		base.ValidDigits.Add("B");
		base.ValidDigits.Add("C");
		base.ValidDigits.Add("D");
		base.ValidDigits.Add("E");
		base.ValidDigits.Add("F");
		base.ValidDigits.Add("G");
		base.ValidDigits.Add("H");
		base.ValidDigits.Add("I");
		base.ValidDigits.Add("J");
		base.ValidDigits.Add("K");
		base.ValidDigits.Add("L");
		base.ValidDigits.Add("M");
		base.ValidDigits.Add("N");
		base.ValidDigits.Add("O");
		base.ValidDigits.Add("P");
		base.ValidDigits.Add("Q");
		base.ValidDigits.Add("R");
		base.ValidDigits.Add("S");
		base.ValidDigits.Add("T");
		base.ValidDigits.Add("U");
		base.ValidDigits.Add("V");
		base.ValidDigits.Add("W");
		base.ValidDigits.Add("Z");
	}

	public override void SetValidPositions()
	{
		base.ValidPositions = new List<Coordinates>();
		base.ValidPositions.Add(new Coordinates() { Position = "A", Row = 3, Column = 5 });
		base.ValidPositions.Add(new Coordinates() { Position = "B", Row = 3, Column = 4 });
		base.ValidPositions.Add(new Coordinates() { Position = "C", Row = 3, Column = 3 });
		base.ValidPositions.Add(new Coordinates() { Position = "D", Row = 3, Column = 2 });
		base.ValidPositions.Add(new Coordinates() { Position = "E", Row = 3, Column = 1 });
		base.ValidPositions.Add(new Coordinates() { Position = "F", Row = 3, Column = 0 });
		base.ValidPositions.Add(new Coordinates() { Position = "G", Row = 2, Column = 5 });
		base.ValidPositions.Add(new Coordinates() { Position = "H", Row = 2, Column = 4 });
		base.ValidPositions.Add(new Coordinates() { Position = "I", Row = 2, Column = 3 });
		base.ValidPositions.Add(new Coordinates() { Position = "J", Row = 2, Column = 2 });
		base.ValidPositions.Add(new Coordinates() { Position = "K", Row = 2, Column = 1 });
		base.ValidPositions.Add(new Coordinates() { Position = "L", Row = 2, Column = 0 });
		base.ValidPositions.Add(new Coordinates() { Position = "M", Row = 1, Column = 5 });
		base.ValidPositions.Add(new Coordinates() { Position = "N", Row = 1, Column = 4 });
		base.ValidPositions.Add(new Coordinates() { Position = "O", Row = 1, Column = 3 });
		base.ValidPositions.Add(new Coordinates() { Position = "P", Row = 1, Column = 2 });
		base.ValidPositions.Add(new Coordinates() { Position = "Q", Row = 1, Column = 1 });
		base.ValidPositions.Add(new Coordinates() { Position = "R", Row = 1, Column = 0 });
		base.ValidPositions.Add(new Coordinates() { Position = "S", Row = 0, Column = 5 });
		base.ValidPositions.Add(new Coordinates() { Position = "T", Row = 0, Column = 4 });
		base.ValidPositions.Add(new Coordinates() { Position = "U", Row = 0, Column = 3 });
		base.ValidPositions.Add(new Coordinates() { Position = "V", Row = 0, Column = 2 });
		base.ValidPositions.Add(new Coordinates() { Position = "W", Row = 0, Column = 1 });
		base.ValidPositions.Add(new Coordinates() { Position = "Z", Row = 0, Column = 0 });
	}

	public override void SetValidPositionMoviments()
	{
		List<Coordinates> subListA = new List<Coordinates>(); // A=B,J
		subListA.Add(new Coordinates() { Position = "B" });
		subListA.Add(new Coordinates() { Position = "J" });

		List<Coordinates> subListB = new List<Coordinates>(); // B=A,C,E
		subListB.Add(new Coordinates() { Position = "A" });
		subListB.Add(new Coordinates() { Position = "C" });
		subListB.Add(new Coordinates() { Position = "E" });

		List<Coordinates> subListC = new List<Coordinates>(); // C=B,O,B
		subListC.Add(new Coordinates() { Position = "B" });
		subListC.Add(new Coordinates() { Position = "O" });
		subListC.Add(new Coordinates() { Position = "B" });

		List<Coordinates> subListD = new List<Coordinates>(); // D=E,K
		subListD.Add(new Coordinates() { Position = "E" });
		subListD.Add(new Coordinates() { Position = "K" });
		
		List<Coordinates> subListE = new List<Coordinates>(); // E=D,F,H
		subListE.Add(new Coordinates() { Position = "D" });
		subListE.Add(new Coordinates() { Position = "F" });
		subListE.Add(new Coordinates() { Position = "H" });
		subListE.Add(new Coordinates() { Position = "B" });

		List<Coordinates> subListF = new List<Coordinates>(); // F=E,N
		subListF.Add(new Coordinates() { Position = "E" });
		subListF.Add(new Coordinates() { Position = "N" });

		List<Coordinates> subListG = new List<Coordinates>(); // G=H,L
		subListF.Add(new Coordinates() { Position = "H" });
		subListF.Add(new Coordinates() { Position = "L" });

		List<Coordinates> subListH = new List<Coordinates>(); // H=G,E,I
		subListH.Add(new Coordinates() { Position = "G" });
		subListH.Add(new Coordinates() { Position = "E" });
		subListH.Add(new Coordinates() { Position = "I" });	

		List<Coordinates> subListI = new List<Coordinates>(); // I=H,M
		subListI.Add(new Coordinates() { Position = "H" });
		subListI.Add(new Coordinates() { Position = "M" });

		List<Coordinates> subListJ = new List<Coordinates>(); // J=V,A,K
		subListJ.Add(new Coordinates() { Position = "A" });
		subListJ.Add(new Coordinates() { Position = "K" });
		subListJ.Add(new Coordinates() { Position = "V" });

		List<Coordinates> subListK = new List<Coordinates>(); // K=J,L,D,S
		subListK.Add(new Coordinates() { Position = "D" });
		subListK.Add(new Coordinates() { Position = "J" });
		subListK.Add(new Coordinates() { Position = "L" });
		subListK.Add(new Coordinates() { Position = "S" });

		List<Coordinates> subListL = new List<Coordinates>(); // L=G,P,K
		subListL.Add(new Coordinates() { Position = "L" });
		subListL.Add(new Coordinates() { Position = "P" });
		subListL.Add(new Coordinates() { Position = "K" });

		List<Coordinates> subListM = new List<Coordinates>(); // M=I,R,N
		subListM.Add(new Coordinates() { Position = "I" });
		subListM.Add(new Coordinates() { Position = "N" });
		subListM.Add(new Coordinates() { Position = "R" });

		List<Coordinates> subListN = new List<Coordinates>(); // N=F,M,O,U
		subListN.Add(new Coordinates() { Position = "F" });
		subListN.Add(new Coordinates() { Position = "M" });
		subListN.Add(new Coordinates() { Position = "O" });
		subListN.Add(new Coordinates() { Position = "U" });

		List<Coordinates> subListO = new List<Coordinates>(); // O=C,N,Z
		subListO.Add(new Coordinates() { Position = "C" });
		subListO.Add(new Coordinates() { Position = "N" });
		subListO.Add(new Coordinates() { Position = "Z" });

		List<Coordinates> subListP = new List<Coordinates>(); //P = L,Q
		subListP.Add(new Coordinates() { Position = "L" });
		subListP.Add(new Coordinates() { Position = "Q" });

		List<Coordinates> subListQ = new List<Coordinates>(); //Q = P,R,T
		subListQ.Add(new Coordinates() { Position = "P" });
		subListQ.Add(new Coordinates() { Position = "R" });
		subListQ.Add(new Coordinates() { Position = "T" });

		List<Coordinates> subListR = new List<Coordinates>(); //R = M,Q
		subListR.Add(new Coordinates() { Position = "M" });
		subListR.Add(new Coordinates() { Position = "Q" });

		List<Coordinates> subListS = new List<Coordinates>(); //S = K,T
		subListS.Add(new Coordinates() { Position = "K" });
		subListS.Add(new Coordinates() { Position = "T" });

		List<Coordinates> subListT = new List<Coordinates>(); //T = S,Q,U,W
		subListT.Add(new Coordinates() { Position = "S" });
		subListT.Add(new Coordinates() { Position = "Q" });
		subListT.Add(new Coordinates() { Position = "U" });
		subListT.Add(new Coordinates() { Position = "W" });

		List<Coordinates> subListU = new List<Coordinates>(); //U = N,T
		subListU.Add(new Coordinates() { Position = "N" });
		subListU.Add(new Coordinates() { Position = "T" });

		List<Coordinates> subListV = new List<Coordinates>(); //V = J,W
		subListV.Add(new Coordinates() { Position = "J" });
		subListV.Add(new Coordinates() { Position = "W" });

		List<Coordinates> subListW = new List<Coordinates>(); //W = T,Z,V
		subListW.Add(new Coordinates() { Position = "T" });
		subListW.Add(new Coordinates() { Position = "V" });
		subListW.Add(new Coordinates() { Position = "Z" });

		List<Coordinates> subListZ = new List<Coordinates>(); //Z = W,O
		subListZ.Add(new Coordinates() { Position = "W" });
		subListZ.Add(new Coordinates() { Position = "O" });

		base.ValidPositionMoviments = new List<List<Coordinates>>();
		base.ValidPositionMoviments.Add(subListA);
		base.ValidPositionMoviments.Add(subListB);
		base.ValidPositionMoviments.Add(subListC);
		base.ValidPositionMoviments.Add(subListD);
		base.ValidPositionMoviments.Add(subListE);
		base.ValidPositionMoviments.Add(subListF);
		base.ValidPositionMoviments.Add(subListG);
		base.ValidPositionMoviments.Add(subListH);
		base.ValidPositionMoviments.Add(subListI);
		base.ValidPositionMoviments.Add(subListJ);
		base.ValidPositionMoviments.Add(subListK);
		base.ValidPositionMoviments.Add(subListL);
		base.ValidPositionMoviments.Add(subListM);
		base.ValidPositionMoviments.Add(subListN);
		base.ValidPositionMoviments.Add(subListO);
		base.ValidPositionMoviments.Add(subListP);
		base.ValidPositionMoviments.Add(subListQ);
		base.ValidPositionMoviments.Add(subListR);
		base.ValidPositionMoviments.Add(subListS);
		base.ValidPositionMoviments.Add(subListT);
		base.ValidPositionMoviments.Add(subListU);
		base.ValidPositionMoviments.Add(subListV);
		base.ValidPositionMoviments.Add(subListW);
		base.ValidPositionMoviments.Add(subListZ);
	}

	private Coordinates GetCoordinatesOfAPiece(Piece piece)
	{
		Coordinates positionToBeReturned = null;
		foreach (Coordinates position in ValidPositions)
		{
			if (this.BoardItSelf[position.Row, position.Column].Piece.Name == piece.Name
					 && this.BoardItSelf[position.Row, position.Column].Piece.FromWhichPlayer.Name == piece.FromWhichPlayer.Name
					 && this.BoardItSelf[position.Row, position.Column].Piece.Symbol == piece.Symbol)
				positionToBeReturned = position;
		}

		return positionToBeReturned;
	}

	private string GetPrintLine1()
	{
		string position1 = base.BoardItSelf[3, 5].Piece.Symbol + "____";
		position1 = "[" + position1.Substring(0, 5)+ "]";
		position1 += "-----------------";

		string position2 = base.BoardItSelf[3, 4].Piece.Symbol + "____";
		position2 = "[" + position2.Substring(0, 5) + "]";
		position2 += "----------------";

		string position3 = base.BoardItSelf[3, 3].Piece.Symbol + "____";
		position3 = "[" + position3.Substring(0, 5) + "]";

		return (position1 + position2 + position3);
	}

	private string GetPrintLine2()
	{
		string val1 = base.BoardItSelf[3, 2].Piece.Symbol + "____";
		val1 = "[" + val1.Substring(0, 5) + "]";

		string val2 = base.BoardItSelf[3, 1].Piece.Symbol + "____";
		val2 = "[" + val2.Substring(0, 5) + "]";

		string val3 = base.BoardItSelf[3, 0].Piece.Symbol + "____";
		val3 = "[" + val3.Substring(0, 5) + "]";

		string position1 = "\n    |                      |                        |";
		position1 += "\n    |   " + val1;
		position1 += "---------";

		string position2 = val2;
		position2+= "----------";

		string position3 = val3 + "    |";

		return (position1 + position2 + position3);
	}

	private string GetPrintLine3()
	{
		string val1 = base.BoardItSelf[2, 5].Piece.Symbol + "____";
		val1 = "[" + val1.Substring(0, 5) + "]";

		string val2 = base.BoardItSelf[2, 4].Piece.Symbol + "____";
		val2 = "[" + val2.Substring(0, 5) + "]";

		string val3 = base.BoardItSelf[2, 3].Piece.Symbol + "____";
		val3 = "[" + val3.Substring(0, 5) + "]";

		string position1 = "\n    |     |                |                |       |";
		position1 += "\n    |     |    " + val1;
		position1 += "--";

		string position2 = val2;
		position2 += "--";

		string position3 = val3 + "    |       |";

		return (position1 + position2 + position3);
	}

	private string GetPrintLine4()
	{
		string val1 = base.BoardItSelf[2, 2].Piece.Symbol + "____";
		val1 = "[" + val1.Substring(0, 5) + "]";

		string val2 = base.BoardItSelf[2, 1].Piece.Symbol + "____";
		val2 = "[" + val2.Substring(0, 5) + "]";

		string val3 = base.BoardItSelf[2, 0].Piece.Symbol + "____";
		val3 = "[" + val3.Substring(0, 5) + "]";

		string val4 = base.BoardItSelf[1, 5].Piece.Symbol + "____";
		val4 = "[" + val4.Substring(0, 5) + "]";

		string val5 = base.BoardItSelf[1, 4].Piece.Symbol + "____";
		val5 = "[" + val5.Substring(0, 5) + "]";

		string val6 = base.BoardItSelf[1, 3].Piece.Symbol + "____";
		val6 = "[" + val6.Substring(0, 5) + "]";

		string position1 = "\n    |     |       |                  |      |       |";
		position1 += "\n" + val1;
		//position1 += "-";

		string position2 = " " + val2;
		//position2 += "-";
		
		string position3 = val3;
		position3 += "           ";

		string position4 = val4;
		//position4 += "-";

		string position5 = val5;
		//position5 += "-";

		string position6 = val6;

		return (position1 + position2 + position3 + position4 + position5 + position6);
	}

	private string GetPrintLine5()
	{
		string val1 = base.BoardItSelf[1, 2].Piece.Symbol + "____";
		val1 = "[" + val1.Substring(0, 5) + "]";

		string val2 = base.BoardItSelf[1, 1].Piece.Symbol + "____";
		val2 = "[" + val2.Substring(0, 5) + "]";

		string val3 = base.BoardItSelf[1, 0].Piece.Symbol + "____";
		val3 = "[" + val3.Substring(0, 5) + "]";

		string position1 = "\n    |     |       |                  |      |       |";
		position1 += "\n    |     |    " + val1;
		position1 += "--";

		string position2 = val2;
		position2 += "--";

		string position3 = val3 + "    |       |";

		return (position1 + position2 + position3);
	}

	private string GetPrintLine6()
	{
		string val1 = base.BoardItSelf[0, 5].Piece.Symbol + "____";
		val1 = "[" + val1.Substring(0, 5) + "]";

		string val2 = base.BoardItSelf[0, 4].Piece.Symbol + "____";
		val2 = "[" + val2.Substring(0, 5) + "]";

		string val3 = base.BoardItSelf[0, 3].Piece.Symbol + "____";
		val3 = "[" + val3.Substring(0, 5) + "]";

		string position1 = "\n    |     |                |                |       |";
		position1 += "\n    |   " + val1;
		position1 += "---------";

		string position2 = val2;
		position2 += "---------";

		string position3 = val3 + "     |";
		position3 += "\n    |                      |                        |";

		return (position1 + position2 + position3);
	}

	private string GetPrintLine7()
	{
		string position1 = base.BoardItSelf[0, 2].Piece.Symbol + "____";
		position1 = "\n[" + position1.Substring(0, 5) + "]";
		position1 += "-----------------";

		string position2 = base.BoardItSelf[0, 1].Piece.Symbol + "____";
		position2 = "[" + position2.Substring(0, 5) + "]";
		position2 += "----------------";

		string position3 = base.BoardItSelf[0, 0].Piece.Symbol + "____";
		position3 = "[" + position3.Substring(0, 5) + "]";

		return (position1 + position2 + position3);
	}

	private void ConfirmMill()
	{
		Piece piece1 = new Piece();
		Piece piece2 = new Piece();
		Piece piece3 = new Piece();

		if (this.IsAnySequenceMill(out piece1, out piece2, out piece3))
		{
			base.PrintConfirmedMill();
			base.SetMills(piece1, piece2, piece3);

			Piece piece = new Piece();
			piece = base.GetPiceOtherPlayer();

			while(piece==null)
			{
				this.PrintPhysicalBoard();
				base.PrintChooseAPieceToBeRemoved();
				piece = base.GetPiceOtherPlayer();
			}

			Coordinates oldCoordinate = this.GetCoordinatesOfAPiece(piece);
			base.RemovePiece(piece);
			base.SetPositionAsNoPlayed(oldCoordinate);			
			this.PrintPhysicalBoard();
		}
	}

	private bool IsPieceNoInAnyMill(Piece piece)
	{
		string v = base.GetListPiecesAvailablePlayer1();

		if (base.Mills != null)
		{
			foreach (var value in Mills)
			{
				foreach (Piece p in value)
				{
					if (p.Name == piece.Name
					 && p.FromWhichPlayer == piece.FromWhichPlayer)
					{
						if(piece.FromWhichPlayer == base.Player1)
						{
							if (base.IsAnyPieceAvailablePlayer1())
								return false;
						}

						if (piece.FromWhichPlayer == base.Player2)
						{
							if (base.IsAnyPieceAvailablePlayer2())
								return false;
						}

						return true;
					}
						
				}
			}
		}

		return true;
	}

	private bool IsMillCompleted(Piece piece1, Piece piece2, Piece piece3)
	{
		int i = 0;

		if (base.Mills != null)
		{
			foreach (var value in base.Mills)
			{
				
				foreach (Piece piece in value)
				{
					if (piece.Name == piece1.Name
					 && piece.FromWhichPlayer == piece1.FromWhichPlayer)
						i++;

					if (piece.Name == piece2.Name
					 && piece.FromWhichPlayer == piece2.FromWhichPlayer)
						i++;

					if (piece.Name == piece3.Name
					 && piece.FromWhichPlayer == piece3.FromWhichPlayer)
						i++;
				}

			}

		}

		return (i >=3) ? true : false;
	}

	private bool IsAnySequenceMill(out Piece piece1
							     , out Piece piece2
							     , out Piece piece3)
	{
		return (IsSequenceMill1(out piece1, out piece2, out piece3)
			 || IsSequenceMill2(out piece1, out piece2, out piece3)
			 || IsSequenceMill3(out piece1, out piece2, out piece3)
			 || IsSequenceMill4(out piece1, out piece2, out piece3)
			 || IsSequenceMill5(out piece1, out piece2, out piece3)
			 || IsSequenceMill6(out piece1, out piece2, out piece3)
			 || IsSequenceMill7(out piece1, out piece2, out piece3)
			 || IsSequenceMill8(out piece1, out piece2, out piece3)
			 || IsSequenceMill9(out piece1, out piece2, out piece3)
			 || IsSequenceMill10(out piece1, out piece2, out piece3)
			 || IsSequenceMill11(out piece1, out piece2, out piece3)
			 || IsSequenceMill12(out piece1, out piece2, out piece3)
			 || IsSequenceMill13(out piece1, out piece2, out piece3)
			 || IsSequenceMill14(out piece1, out piece2, out piece3)
			 || IsSequenceMill15(out piece1, out piece2, out piece3)
			 || IsSequenceMill16(out piece1, out piece2, out piece3)) ? true : false;
	}

	private bool IsSequenceMill1(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{

		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[3, 5].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[3, 4].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[3, 3].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[3, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[3, 4].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[3, 3].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 4].Piece.FromWhichPlayer.Name == this.BoardItSelf[3, 3].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[3, 5].Piece;
			piece2 = this.BoardItSelf[3, 4].Piece;
			piece3 = this.BoardItSelf[3, 3].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}
			
		return false;
	}

	private bool IsSequenceMill2(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[3, 5].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[2, 2].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 2].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[3, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 2].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 2].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 2].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[3, 5].Piece;
			piece2 = this.BoardItSelf[2, 2].Piece;
			piece3 = this.BoardItSelf[0, 2].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill3(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[3, 4].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[3, 1].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[2, 4].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[3, 4].Piece.FromWhichPlayer.Name == this.BoardItSelf[3, 1].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 4].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 4].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 1].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 4].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[3, 4].Piece;
			piece2 = this.BoardItSelf[3, 1].Piece;
			piece3 = this.BoardItSelf[2, 4].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill4(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[3, 3].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 3].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 0].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[3, 3].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 3].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 3].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 0].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[1, 3].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 0].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[3, 3].Piece;
			piece2 = this.BoardItSelf[1, 3].Piece;
			piece3 = this.BoardItSelf[0, 0].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill5(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[3, 2].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[3, 1].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[3, 0].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[3, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[3, 1].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[3, 0].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 1].Piece.FromWhichPlayer.Name == this.BoardItSelf[3, 0].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[3, 2].Piece;
			piece2 = this.BoardItSelf[3, 1].Piece;
			piece3 = this.BoardItSelf[3, 0].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill6(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[3, 2].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[2, 1].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 5].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[3, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 1].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 5].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 1].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 5].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[3, 2].Piece;
			piece2 = this.BoardItSelf[2, 1].Piece;
			piece3 = this.BoardItSelf[0, 5].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill7(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[3, 0].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 4].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 3].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[3, 0].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 4].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[3, 0].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 3].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[1, 4].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 3].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[3, 0].Piece;
			piece2 = this.BoardItSelf[1, 4].Piece;
			piece3 = this.BoardItSelf[0, 3].Piece;

			return true;
		}

		if (!this.IsMillCompleted(piece1, piece2, piece3))
			return true;

		return false;
	}

	private bool IsSequenceMill8(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[2, 5].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[2, 3].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[2, 4].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[2, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 4].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 3].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 4].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 3].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[2, 5].Piece;
			piece2 = this.BoardItSelf[2, 3].Piece;
			piece3 = this.BoardItSelf[2, 4].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill9(out Piece piece1
							   , out Piece piece2
							   , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[2, 5].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[2, 0].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 2].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[2, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 0].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 2].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 0].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 2].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[2, 5].Piece;
			piece2 = this.BoardItSelf[2, 0].Piece;
			piece3 = this.BoardItSelf[1, 2].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill10(out Piece piece1
							    , out Piece piece2
							    , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[2, 3].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 0].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 5].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[2, 3].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 5].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 3].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 0].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[1, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 0].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[2, 3].Piece;
			piece2 = this.BoardItSelf[1, 0].Piece;
			piece3 = this.BoardItSelf[1, 5].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill11(out Piece piece1
							    , out Piece piece2
							    , out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[2, 2].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[2, 1].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[2, 0].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[2, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 1].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 0].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[2, 1].Piece.FromWhichPlayer.Name == this.BoardItSelf[2, 0].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[2, 2].Piece;
			piece2 = this.BoardItSelf[2, 1].Piece;
			piece3 = this.BoardItSelf[2, 0].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill12(out Piece piece1
								, out Piece piece2
								, out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[1, 5].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 3].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 4].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[1, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 4].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[1, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 3].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[1, 4].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 3].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[1, 5].Piece;
			piece2 = this.BoardItSelf[1, 3].Piece;
			piece3 = this.BoardItSelf[1, 4].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill13(out Piece piece1
								, out Piece piece2
								, out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[1, 2].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 1].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[1, 0].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[1, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 1].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[1, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 0].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[1, 1].Piece.FromWhichPlayer.Name == this.BoardItSelf[1, 0].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[1, 2].Piece;
			piece2 = this.BoardItSelf[1, 1].Piece;
			piece3 = this.BoardItSelf[1, 0].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill14(out Piece piece1
								, out Piece piece2
								, out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[1, 1].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 1].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 4].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[1, 1].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 4].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[1, 1].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 1].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[0, 4].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 1].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[1, 1].Piece;
			piece2 = this.BoardItSelf[0, 1].Piece;
			piece3 = this.BoardItSelf[0, 4].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill15(out Piece piece1
								, out Piece piece2
								, out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[0, 5].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 4].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 3].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[0, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 4].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[0, 5].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 3].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[0, 4].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 3].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[0, 5].Piece;
			piece2 = this.BoardItSelf[0, 4].Piece;
			piece3 = this.BoardItSelf[0, 3].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}

	private bool IsSequenceMill16(out Piece piece1
								, out Piece piece2
								, out Piece piece3)
	{
		piece1 = null;
		piece2 = null;
		piece3 = null;

		if (this.BoardItSelf[0, 2].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 1].Piece.FromWhichPlayer == null
		 || this.BoardItSelf[0, 0].Piece.FromWhichPlayer == null)
			return false;

		if (this.BoardItSelf[0, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 1].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[0, 2].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 0].Piece.FromWhichPlayer.Name
		 && this.BoardItSelf[0, 1].Piece.FromWhichPlayer.Name == this.BoardItSelf[0, 0].Piece.FromWhichPlayer.Name)
		{
			piece1 = this.BoardItSelf[0, 2].Piece;
			piece2 = this.BoardItSelf[0, 1].Piece;
			piece3 = this.BoardItSelf[0, 0].Piece;

			if (!this.IsMillCompleted(piece1, piece2, piece3))
				return true;

			return false;
		}

		return false;
	}
		
}
