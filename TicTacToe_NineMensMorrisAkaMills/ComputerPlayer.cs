using System;
using System.Collections.Generic;

public class ComputerPlayer : IPlayer
{
	public ComputerPlayer()
	{
		CoordinatePlayed = new Coordinates();
	}

	public ComputerPlayer(string name, List<Piece> pieces)
	{
		this.Name = name;
		this.Pieces = pieces;
	}

	public string Name
	{ get; set; }
	
	public Coordinates CoordinatePlayed
	{ get; set; }
	
	public List<Piece> Pieces
	{ get; set; }

	public string ChooseAPositionToPlay(List<Coordinates> list)
	{
		return RandomNumber(list).Position;
	}

	private Coordinates RandomNumber(List<Coordinates> list)
	{
		Random random = new Random();
		return list[random.Next(list.Count)];
	}
}
