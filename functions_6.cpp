#include<iostream>
#include<cmath>
#include<armadillo>

std::pair<arma::mat, arma::mat> Node_Division(double x_start, double x_end, double y_start, double y_end, int nx, int ny) {
    arma::vec x = arma::linspace<arma::vec>(x_start, x_end, nx);
    arma::vec y = arma::linspace<arma::vec>(y_start, y_end, ny);

    arma::mat xx = arma::repmat(x, 1, ny);
    arma::mat yy = arma::repmat(y.t(), nx, 1);

    return std::make_pair(xx, yy);
}


std::pair<double, double> Nodal_Coordinates(int nodenumber, double x_start, double x_end, double y_start, double y_end, int nx, int ny) {
    auto coordinates = Node_Division(x_start, x_end, y_start, y_end, nx, ny);
    arma::mat xx = coordinates.first;
    arma::mat yy = coordinates.second;

    int row_number = nodenumber / nx;
    int column_number = nodenumber % nx;

    return std::make_pair(xx(row_number, column_number), yy(row_number, column_number));
}

#include <armadillo>
#include <cmath>

double Distance_Function(const arma::vec &x, int nodenumber, double x_start, double x_end, double y_start, double y_end, int nx, int ny) {
  arma::mat xx, yy;
  Node_Division(x_start, x_end, y_start, y_end, nx, ny, xx, yy);
  int row_number = nodenumber / nx;
  int column_number = nodenumber % nx;
  double xc = xx(row_number, column_number);
  double yc = yy(row_number, column_number);
  return std::sqrt(std::pow(xc - x(0), 2) + std::pow(yc - x(1), 2));
}

double cubic_spline_weight_function(const arma::vec &x, int nodenumber, int nx, int ny, double x_start, double x_end, double y_start, double y_end) {
  double dia_supp = 6 * std::abs(x_end - x_start) / (nx - 1);
  double d = Distance_Function(x, nodenumber, x_start, x_end, y_start, y_end, nx, ny);
  d = 2 * d / (dia_supp);
  if (d <= 0.5) {
    return 2.0 / 3.0 - 4 * std::pow(d, 2) + 4 * std::pow(d, 3);
  } else if (0.5 < d && d <= 1) {
    return 4.0 / 3.0 - 4 * d + 4 * std::pow(d, 2) - (4.0 / 3.0) * std::pow(d, 3);
  } else {
    return 0;
  }
}

double cubic_spline_weight_function_xgradient(const arma::vec &x, int nodenumber, int nx, int ny, double x_start, double x_end, double y_start, double y_end) {
  double d = Distance_Function(x, nodenumber, x_start, x_end, y_start, y_end, nx, ny);
  double dia_supp = 6 * std::abs(x_end - x_start) / (nx - 1);
  double rw = dia_supp / 2;
  double D = 2 * d / (dia_supp);
  if (D <= 0.5) {
    return (-8 * D + 12 * std::pow(D, 2)) * (1 / rw) * (x(0) / d);
  } else if (0.5 < D && D <= 1) {
    return (-4 + 8 * D - 4 * std::pow(D, 2)) * (1 / rw) * (x(0) / d);
  } else {
    return 0;
  }
}
