using System;
using System.Collections.Generic;

public class Players
{
	private readonly Dictionary<TypeOfPlayer, PlayerFactory> _factories;

	public Players()
	{
		_factories = new Dictionary<TypeOfPlayer, PlayerFactory>();

		foreach (TypeOfPlayer typeOfPlayer in Enum.GetValues(typeof(TypeOfPlayer)))
		{

			var factory = (PlayerFactory)Activator.CreateInstance(Type.GetType(Enum.GetName(typeof(TypeOfPlayer), typeOfPlayer) + "PlayerFactory"));
			_factories.Add(typeOfPlayer, factory);
		}

	}

	public IPlayer ExecuteCreation(TypeOfPlayer typeOfPlayer, string name, List<Piece> pieces) => _factories[typeOfPlayer].Create(name, pieces);
}
