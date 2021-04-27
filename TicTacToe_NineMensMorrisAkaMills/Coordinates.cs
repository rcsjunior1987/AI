using System;

public class Coordinates
{
    public string Position
    { get; set; }

    public int Row
    { get; set; }

    public int Column
    { get; set; }

    public Coordinates()
    {

    }

    public Coordinates(string Position, int row, int column)
    {
        this.Position = Position;
		this.Row = row;
		this.Column = column;
		//this.SetPositionToRowColumn();
	}

	/*private void SetPositionToRowColumn()
	{
		switch (this.Position)
		{
			case "1":
				Row = 0;
				Column = 0;
				break;
			case "2":
				Row = 0;
				Column = 1;
				break;
			case "3":
				Row = 0;
				Column = 2;
				break;
			case "4":
				Row = 1;
				Column = 0;
				break;
			case "5":
				Row = 1;
				Column = 1;
				break;
			case "6":
				Row = 1;
				Column = 2;
				break;
			case "7":
				Row = 2;
				Column = 0;
				break;
			case "8":
				Row = 2;
				Column = 1;
				break;
			case "9":
				Row = 2;
				Column = 2;
				break;
		}
	}*/
}
