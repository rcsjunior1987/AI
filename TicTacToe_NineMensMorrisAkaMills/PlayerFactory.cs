using System;
using System.Collections.Generic;

public abstract class PlayerFactory
{
	public PlayerFactory()
	{

	}

	public abstract IPlayer Create(string name, List<Piece> pieces);
}
