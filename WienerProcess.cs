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
    public static class WienerProcess
    {
        /// <summary>
        /// Generuje proces Wienerowski 2D przy użyciu Infer.NET i (opcjonalnie) zapisuje wykres.
        /// </summary>
        /// <param name="outputPath">Ścieżka do pliku PNG.</param>
        /// <param name="steps">Liczba kroków.</param>
        /// <param name="totalTime">Całkowity czas trwania procesu.</param>
        /// <param name="exportPlot">Czy zapisać wykres jako PNG.</param>
        public static void GenerateAndSave(string outputPath, int steps = 2500, double totalTime = 1.0, bool exportPlot = true)
        {
            double dt = totalTime / steps;

            //zakres czasowy
            Variable<int> numSteps = Variable.Observed(steps);
            Range time = new Range(numSteps);

            //dwie niezależne zmienne losowe
            var x = Variable.Array<double>(time);
            var y = Variable.Array<double>(time);

            //definicja procesu Wienerowskiego
            using (var t = Variable.ForEach(time))
            {
                var i = t.Index;

                using (Variable.If(i == 0))
                {
                    //warunki początkowe i constraint rozbijający symetrię
                    x[i] = Variable.GaussianFromMeanAndVariance(0, 1);
                    y[i] = Variable.GaussianFromMeanAndVariance(0, 1);
                    Variable.ConstrainEqual(x[i], Variable.Constant(1.0));
                    Variable.ConstrainEqual(y[i], Variable.Constant(1.0));
                }

                using (Variable.If(i > 0))
                {
                    x[i] = Variable.GaussianFromMeanAndVariance(x[i - 1], dt);
                    y[i] = Variable.GaussianFromMeanAndVariance(y[i - 1], dt);
                }
            }

            var engine = new InferenceEngine();
            var posteriorX = engine.Infer<Gaussian[]>(x);
            var posteriorY = engine.Infer<Gaussian[]>(y);

            //ścieżka z wartości oczekiwanych
            var points = GeneratePathFromPosterior(posteriorX, posteriorY, steps, scale: 1000.0);
            /*
            Console.WriteLine($"[WienerProcess] Kroków: {points.Count}");
            Console.WriteLine($"[WienerProcess] Start: ({points[0].X:0.###}, {points[0].Y:0.###})");
            Console.WriteLine($"[WienerProcess] Koniec: ({points[^1].X:0.###}, {points[^1].Y:0.###})");
            Console.WriteLine($"[WienerProcess] Mean końcowy X: {posteriorX[^1].GetMean():0.####}, Y: {posteriorY[^1].GetMean():0.####}");
            */
            //eksport wykresu jeśli ustawione
            if (exportPlot)
            {
                var model = CreatePlotModel(points);
                PngExporter.Export(model, outputPath, 800, 800, 96);
                Console.WriteLine($"Wykres zapisany: {Path.GetFullPath(outputPath)}");
            }
        }

        /// <summary>
        /// Tworzy listę punktów z wartości oczekiwanych posteriorów.
        /// </summary>
        private static List<DataPoint> GeneratePathFromPosterior(Gaussian[] posteriorX, Gaussian[] posteriorY, int steps, double scale)
        {
            var points = new List<DataPoint>();
            double x = 0, y = 0;
            points.Add(new DataPoint(x, y));

            for (int i = 1; i < steps; i++)
            {
                double dx = (posteriorX[i].GetMean() - posteriorX[i - 1].GetMean()) * scale;
                double dy = (posteriorY[i].GetMean() - posteriorY[i - 1].GetMean()) * scale;

                x += dx;
                y += dy;
                points.Add(new DataPoint(x, y));
            }

            return points;
        }

        /// <summary>
        /// Tworzy wykres OxyPlot bez osi, z prostą czarną linią.
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