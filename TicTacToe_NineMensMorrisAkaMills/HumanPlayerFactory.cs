using System;
using System.Collections.Generic;

public class HumanPlayerFactory : PlayerFactory
{
	public HumanPlayerFactory()
	{
	}

	public override IPlayer Create(string name, List<Piece> pieces) => new HumanPlayer(name, pieces);

}
