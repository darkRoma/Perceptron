using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Windows;




namespace Perceptron
{
    public partial class Form1 : Form
    {

        public Graphics g;
        public Pen brush;

        public Point drawPoint;
        public bool needToDraw;

        public Bitmap flag;

        public Point Border1;
        public Point Border2;

        List<int> list;      
        
        public Form1()
        {
            InitializeComponent();

            list = new List<int>();
  
            flag = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            g = Graphics.FromImage(flag);            
            //g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.Clear(Color.White);
            
            brush = new Pen(Color.Black,6f);
            needToDraw = false;
        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            needToDraw = true;
            drawPoint = e.Location;
        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            if (needToDraw)
            {
                g.DrawLine(brush, drawPoint, e.Location);
                
                drawPoint = e.Location;
            }
            pictureBox1.Image = flag;
        }

        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
            needToDraw = false;                      
        }

        public void getBorders()
        {
            Border1.X = -1; Border1.Y = pictureBox1.Height;
            Border2.X = -1; Border2.Y = 0;

            for (int i = 0; i < pictureBox1.Width; i++)
            {
                for (int j = 0; j < pictureBox1.Height; j++)
                {
                    Color tempColor = Color.FromArgb(255, 0, 0,0);
                   
                    if (flag.GetPixel(i,j).Equals(tempColor))
                    {
                       if (Border1.X==-1) Border1.X = i;
                       Border1.Y = Math.Min(j, Border1.Y);
                       Border2.X = i;
                       Border2.Y = Math.Max(Border2.Y, j);
                    }
                }
            }
            g.DrawRectangle(Pens.Blue, new Rectangle(Border1.X, Border1.Y, Border2.X-Border1.X,Border2.Y-Border1.Y));
        }

        public void getVector()
        {
            int countOfRectangles = 10;

            System.Drawing.SolidBrush myBrush = new System.Drawing.SolidBrush(System.Drawing.Color.Black);

            for (int i=0; i<countOfRectangles; i++)
                for (int j = 0; j < countOfRectangles; j++)
                {
                    int dx = (int)Math.Round( (double)((Border2.X - Border1.X) / countOfRectangles) );
                    int dy = (int)Math.Round((double)((Border2.Y - Border1.Y) / countOfRectangles));
                    int dxRest=0, dyRest=0;

                    bool isBlackPixelInside = false;
                    Color tempColor = Color.FromArgb(255, 0, 0, 0);

                    if (j == countOfRectangles - 1)
                        dxRest = Border2.X - (Border1.X + countOfRectangles * dx);
                    if (i == countOfRectangles - 1) 
                        dyRest = Border2.Y - (Border1.Y + countOfRectangles * dy);

                    for (int y = Border1.Y + i * dy; y <= (Border1.Y + (i + 1) * dy + dyRest); y++)
                    {
                        for (int x = Border1.X + j * dx; x <= (Border1.X + (j + 1) * dx + dxRest); x++)
                        {
                            if (flag.GetPixel(x, y).Equals(tempColor)) isBlackPixelInside = true;
                        }
                        if (isBlackPixelInside) break;
                    }
                    if (isBlackPixelInside)
                    {
                        list.Add(1);
                        g.FillRectangle(myBrush, new Rectangle(Border1.X + j * dx, Border1.Y + i * dy, dx + dxRest, dy + dyRest));
                    }
                    else list.Add(0);
                }
            list.Add(-1);

            
            
            myBrush.Dispose();

        }

        public void printVectorToFile()
        {
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(@".\letterA.txt", true))
            {
                for (int i = 0; i < list.Count; i++) file.Write(list.ElementAt(i)+" ");
                file.WriteLine();
            }
        }


        private void button1_Click(object sender, EventArgs e)
        {
            list.Clear();

            getBorders();

            getVector();

            printVectorToFile();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            g.Clear(Color.White);
        }

        public void tryToMakeForwardPass()
        {
            NeuralNetwork network = new NeuralNetwork();

            List<List<int>> listsFromFile = new List<List<int>>();

            int[] arr = System.IO.File.ReadAllText(@".\letterA.txt").Split(' ').Select(n => int.Parse(n)).ToArray();

            int c=0;
            while (c != arr.Length)
            {
                List<int> tempList = new List<int>(); ;
                for (int i = 0; i < 101; i++)
                {
                    tempList.Add(arr[i]);
                    c++;
                }
                listsFromFile.Add(tempList);
            }

            network.learningList = listsFromFile;

            network.initializeOutputs();

            network.forwardPass();

            double someVar = network.currentError;

            MessageBox.Show(someVar.ToString());
        }

        private void button3_Click(object sender, EventArgs e)
        {
            tryToMakeForwardPass();
        }

    }
}
