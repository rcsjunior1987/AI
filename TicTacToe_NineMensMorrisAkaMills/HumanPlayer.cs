using System;
using System.Collections.Generic;

public class HumanPlayer : IPlayer
{
	public string Name
	{ get; set; }
	
	public Coordinates CoordinatePlayed
	{ get; set; }
	public List<Piece> Pieces
	{ get; set; }

	public HumanPlayer()
	{
		CoordinatePlayed = new Coordinates();
	}

	public HumanPlayer(string name, List<Piece> pieces)
	{
		this.Name = name;
		this.Pieces = pieces;
	}

	public string ChooseAPositionToPlay(List<Coordinates> list)
	{
		string a = "";
		a = Console.ReadLine();
		return a;

	}
}

