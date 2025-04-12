using System;
using System.Collections.Generic;
using System.IO;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.ImageSharp;

namespace PathVisualizer
{
    public static class MultiPathVisualizer
    {
        //typy wspieranych ścieżek 2D
        public enum PathType
        {
            RandomWalk, //skok jednostkowy w 4 kierunkach (góra/dół/lewo/prawo)
            Brownian, //N(0, sqrt(dt))
            Wiener, // N(0, dt)
            GaussianRandomWalk
        }

        /// <summary>
        /// Główna metoda – generuje i zapisuje wykres z wieloma ścieżkami
        /// </summary>
        /// <param name="outputPath">Ścieżka do pliku PNG</param>
        /// <param name="numPaths">Liczba ścieżek</param>
        /// <param name="steps">Liczba kroków każdej ścieżki</param>
        /// <param name="type">Typ ścieżki</param>
        public static void GenerateAndSave(string outputPath, int numPaths = 5, int steps = 1000, PathType type = PathType.GaussianRandomWalk)
        {
            var model = new PlotModel
            {
                Background = OxyColors.White,
                PlotAreaBorderThickness = new OxyThickness(0) //bez obramowania
            };

            var rand = new Random();
            var palette = OxyPalettes.HueDistinct(numPaths); //kolor dla każdej ścieżki

            //generowanie ścieżek
            for (int i = 0; i < numPaths; i++)
            {
                var points = GeneratePath(rand, steps, type);

                var series = new LineSeries
                {
                    Color = palette.Colors[i], //różny kolor
                    StrokeThickness = 1,
                    LineStyle = LineStyle.Solid,
                    MarkerType = MarkerType.None
                };

                series.Points.AddRange(points);
                model.Series.Add(series);
            }

            model.Axes.Clear(); //brak osi

            PngExporter.Export(model, outputPath, 800, 800, 96);
            Console.WriteLine($"Zapisano wykres wielu ścieżek ({type}): {Path.GetFullPath(outputPath)}");
        }

        /// <summary>
        /// Generuje jedną ścieżkę 2D określonego typu
        /// </summary>
        private static List<DataPoint> GeneratePath(Random rand, int steps, PathType type)
        {
            var points = new List<DataPoint>();
            double x = 0, y = 0;
            double dt = 1.0 / steps;

            points.Add(new DataPoint(x, y)); //punkt startowy

            for (int i = 0; i < steps; i++)
            {
                switch (type)
                {
                    case PathType.RandomWalk:
                        //ruch o 1 w jednym z 4 kierunków
                        int direction = rand.Next(4);
                        switch (direction)
                        {
                            case 0: y += 1; break; //góra
                            case 1: x += 1; break; //prawo
                            case 2: y -= 1; break; //dół
                            case 3: x -= 1; break; //lewo
                        }
                        break;

                    case PathType.Brownian:
                        //przesunięcie ~ N(0, sqrt(dt))
                        x += Math.Sqrt(dt) * SampleStandardNormal(rand);
                        y += Math.Sqrt(dt) * SampleStandardNormal(rand);
                        break;

                    case PathType.Wiener:
                        //przesunięcie ~ N(0, dt)
                        x += SampleNormal(rand, 0, Math.Sqrt(dt));
                        y += SampleNormal(rand, 0, Math.Sqrt(dt));
                        break;
                    case PathType.GaussianRandomWalk:
                        x += SampleNormal(rand, 0, 1);
                        y += SampleNormal(rand, 0, 1);
                    break;
                }

                points.Add(new DataPoint(x, y));
            }

            return points;
        }

        /// <summary>
        /// Próbka z rozkładu normalnego N(0,1)
        /// </summary>
        private static double SampleStandardNormal(Random rand)
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        /// <summary>
        /// Próbka z rozkładu normalnego N(mu, sigma)
        /// </summary>
        private static double SampleNormal(Random rand, double mean, double stddev)
        {
            return mean + stddev * SampleStandardNormal(rand);
        }
    }
}