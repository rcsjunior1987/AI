using System;

public class Piece
{	
	public Piece()
	{
	}

	public string Name
	{ get; set; }

	public string Symbol
	{ get; set; }

	public IPlayer FromWhichPlayer
	{ get; set; }
}
