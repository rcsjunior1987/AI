using System;
using System.Collections.Generic;

public interface IBoardGame
{
	bool PlayItSelf();
	bool IsTheCurrentPlayerTheWinner();
	void SetValidDigits();
	void SetValidPositions();
	bool PrintPhysicalBoard();
	void SetPiecesPlayer1(List<Piece> Pieces);
	void SetPiecesPlayer2(List<Piece> Pieces);
	List<Piece> GetPiecesPlayer1();
	List<Piece> GetPiecesPlayer2();
	bool InitGame();
	void StartNewGame();
	void SaveItSelf();
	void CancelItSelf();
	string GetDigit();
}
