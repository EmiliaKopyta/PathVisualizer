using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.ImageSharp;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace PathVisualizer
{
    public static class Markov2D
    {
        /// <summary>
        /// Generuje trajektorię 2D z modelu siatki Markowa i (opcjonalnie) zapisuje wykres do pliku.
        /// </summary>
        public static void GenerateAndSave(string outputPath, int size = 30, double step = 0.2, bool exportPlot = true)
        {
            //definicja siatki Markowa
            Range rows = new Range(size);
            Range cols = new Range(size);
            VariableArray2D<double> states = Variable.Array<double>(rows, cols);

            using (var rowBlock = Variable.ForEach(rows))
            {
                var row = rowBlock.Index;
                using (var colBlock = Variable.ForEach(cols))
                {
                    var col = colBlock.Index;

                    using (Variable.If(row == 0))
                    {
                        states[row, col] = Variable.GaussianFromMeanAndVariance(0, 1);
                    }
                    using (Variable.If(row > 0))
                    {
                        states[row, col] = Variable.GaussianFromMeanAndVariance(states[row - 1, col], 1);
                    }

                    using (Variable.If(col > 0))
                    {
                        Variable.ConstrainEqualRandom(states[row, col] - states[row, col - 1], new Gaussian(0, 1));
                    }
                }
            }

            //constraint rozbijający symetrię w (0,0)
            Variable.ConstrainEqual(states[0, 0], Variable.Constant(10.0));

            var engine = new InferenceEngine();
            var posterior = engine.Infer<Gaussian[,]>(states);

            //średnia i odchylenie
            double sum = 0, sumSq = 0;
            int total = size * size;

            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                {
                    double m = posterior[i, j].GetMean();
                    sum += m;
                    sumSq += m * m;
                }

            double meanAvg = sum / total;
            double stddev = Math.Sqrt(sumSq / total - meanAvg * meanAvg);

            Console.WriteLine($"[Markov2D] Mean: {meanAvg:0.###}, Stddev: {stddev:0.###}");

            //budowanie ścieżki
            double x = 0, y = 0;
            var points = new List<DataPoint> { new DataPoint(x, y) };

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    double m = posterior[i, j].GetMean();
                    double z = (m - meanAvg) / stddev;

                    if (z > step)
                        x += 0.2; //silnie dodatni — krok w prawo
                    else if (z < -step)
                        x -= 0.2; //silnie ujemny — krok w lewo
                    else
                        y += ((i + j) % 2 == 0) ? 0.2 : -0.2; //neutralny — krok pionowo

                    points.Add(new DataPoint(x, y));
                }
            }

            Console.WriteLine($"[Markov2D] Liczba punktów ścieżki: {points.Count}");

            //eksport wykresu
            if (exportPlot)
            {
                var model = CreatePlotModel(points);
                PngExporter.Export(model, outputPath, 1800, 800, 96);
                Console.WriteLine($"Zapisano do pliku: {Path.GetFullPath(outputPath)}");
            }
        }

        /// <summary>
        /// Tworzy model wykresu OxyPlot z podaną listą punktów.
        /// </summary>
        private static PlotModel CreatePlotModel(List<DataPoint> points)
        {
            var model = new PlotModel
            {
                Background = OxyColors.White,
                PlotAreaBorderThickness = new OxyThickness(0)
            };

            var series = new LineSeries
            {
                Color = OxyColors.Black,
                StrokeThickness = 1,
                LineStyle = LineStyle.Solid,
                MarkerType = MarkerType.None
            };

            series.Points.AddRange(points);
            model.Series.Add(series);
            model.Axes.Clear();

            return model;
        }
    }
}