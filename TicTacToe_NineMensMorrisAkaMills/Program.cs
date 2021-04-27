using System;

class Program
{

    static void Main(string[] args)
    {
        ChooseGameToBePlayed();
    }

    private static void ChooseGameToBePlayed()
    {
        BoardGame gameChosen = new BoardGame();
        gameChosen = gameChosen.ChooseGameToBePlayed();

        while (gameChosen.InitGame())
        {
            gameChosen = new BoardGame();
            gameChosen = gameChosen.ChooseGameToBePlayed();
        }

        ChooseGameToBePlayed();
    }

}
