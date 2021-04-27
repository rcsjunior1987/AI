using System;
using System.Collections.Generic;

public class ComputerPlayerFactory : PlayerFactory
{
	public ComputerPlayerFactory()
	{
	}
	 
	public override IPlayer Create(string name, List<Piece> pieces) => new ComputerPlayer(name, pieces);
}
