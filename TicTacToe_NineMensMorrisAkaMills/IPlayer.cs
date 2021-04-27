using System;
using System.Collections.Generic;

public interface IPlayer
{
	string Name
	{ get; set; }
	List<Piece> Pieces
	{ get; set; }
	string ChooseAPositionToPlay(List<Coordinates> list);
	Coordinates CoordinatePlayed
	{ get; set; }
}
