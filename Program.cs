using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using Microsoft.ML.Probabilistic.Math;
using OxyPlot.ImageSharp;
using OxyPlot.Series;
using OxyPlot;
using SixLabors.ImageSharp.Processing;
using PathVisualizer;

class BernoulliInference
{
    static void Main()
    {
        //miałam problem z wykonaniem wizualizacji na podstawie kodu wykonanego z Infer .NET,
        //jedynie Markov grid ma dość udaną wizualizację
        //żeby zobaczyć uproszczone, ale znacznie lepsze wizualizacje, stworzyłam klasę MultiPathVisualizer
        //1) Markov chain
        Markov1D.GenerateAndSave("markov_1d.png", 100, false);

        //2) Markov grid
        Markov2D.GenerateAndSave("markov_2d.png", 100, 0.2, true);

        //3) ruchy Browna
        BrownianMotion.GenerateAndSave("brownian_motion.png", 2500, 1.0, false);

        //4) proces Wienerowski 
        WienerProcess.GenerateAndSave("wiener_process.png", 2500, 1.0, false);

        //wizualizacja kilku ścieżek na jednym wykresie - uproszczona, bez Infer .NET
        MultiPathVisualizer.GenerateAndSave("multi_random_walk.png", 5, 1000, MultiPathVisualizer.PathType.RandomWalk);
        MultiPathVisualizer.GenerateAndSave("multi_brownian.png", 5, 1000, MultiPathVisualizer.PathType.Brownian);
        MultiPathVisualizer.GenerateAndSave("multi_wiener.png", 5, 1000, MultiPathVisualizer.PathType.Wiener);
        MultiPathVisualizer.GenerateAndSave("multi_gaussian_random_walk.png", 5, 1000);

    }
}