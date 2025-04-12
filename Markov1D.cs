using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using OxyPlot.ImageSharp;
using OxyPlot.Series;
using OxyPlot;
using System;
using System.Collections.Generic;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace PathVisualizer
{
    public static class Markov1D
    {
        /// <summary>
        /// Generuje łańcuch Markowa (1D Gaussian Random Walk) i (opcjonalnie) zapisuje wykres.
        /// </summary>
        public static void GenerateAndSave(string outputPath, int steps = 100, bool exportPlot = true)
        {
            //posterior = rozkłady a posteriori dla każdej zmiennej x[t]
            var posterior = GeneratePosterior(steps);

            //wartości oczekiwane z każdego rozkładu
            var means = ExtractMeans(posterior);

            double start = means[0];
            double end = means[^1];
            double avg = 0, sumSq = 0;

            foreach (var m in means)
            {
                avg += m;
                sumSq += m * m;
            }

            avg /= steps;
            double stddev = Math.Sqrt(sumSq / steps - avg * avg);
            /*
            Console.WriteLine($"[Markov1D] Kroków: {steps}");
            Console.WriteLine($"[Markov1D] Start: {start:0.###}, Koniec: {end:0.###}");
            Console.WriteLine($"[Markov1D] Średnia: {avg:0.###}, Odchylenie: {stddev:0.###}");
            */
            if (exportPlot)
            {
                var model = CreatePlotModel(means, "1D Markov Chain (Gaussian Random Walk)");
                PngExporter.Export(model, outputPath, 800, 600, 96);
                Console.WriteLine($"Wykres zapisany: {Path.GetFullPath(outputPath)}");
            }
        }

        /// <summary>
        /// Definicja łańcucha Markowa w Infer.NET: x[0] ~ N(0,1), x[t] ~ N(x[t-1], 1)
        /// </summary>
        private static Gaussian[] GeneratePosterior(int steps)
        {
            Variable<int> numTimes = Variable.Observed(steps);
            Range time = new Range(numTimes);
            VariableArray<double> x = Variable.Array<double>(time);

            using (var block = Variable.ForEach(time))
            {
                var t = block.Index;

                using (Variable.If(t == 0))
                {
                    x[t] = Variable.GaussianFromMeanAndVariance(0, 1);
                    Variable.ConstrainEqual(x[t], Variable.Constant(1.0));
                }

                using (Variable.If(t > 0))
                {
                    x[t] = Variable.GaussianFromMeanAndVariance(x[t - 1], 1);
                }
            }

            var engine = new InferenceEngine();
            return engine.Infer<Gaussian[]>(x);
        }

        /// <summary>
        /// Ekstrakcja wartości oczekiwanych z posteriorów
        /// </summary>
        private static List<double> ExtractMeans(Gaussian[] posterior)
        {
            List<double> means = new List<double>();
            for (int i = 0; i < posterior.Length; i++)
                means.Add(posterior[i].GetMean());
            return means;
        }

        /// <summary>
        /// Tworzy wykres z linii wartości oczekiwanych
        /// </summary>
        private static PlotModel CreatePlotModel(List<double> means, string title)
        {
            var model = new PlotModel
            {
                Title = title,
                Background = OxyColors.White
            };

            var series = new LineSeries
            {
                Color = OxyColors.Black,
                StrokeThickness = 2,
                LineStyle = LineStyle.Solid
            };

            for (int i = 0; i < means.Count; i++)
            {
                series.Points.Add(new DataPoint(i, means[i]));
            }

            model.Series.Add(series);
            return model;
        }
    }
}