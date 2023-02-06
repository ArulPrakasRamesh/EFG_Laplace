#include "numint.hpp"

Numerical_integration::Numerical_integration(double x_start, double x_end, double y_start, double y_end, int npoints)
{
    this->x_start = x_start;
    this->x_end = x_end;
    this->y_start = y_start;
    this->y_end = y_end;
    this->npoints = npoints;
}

std::pair<arma::mat, arma::mat> Numerical_integration::construct_Delaunay()
{
    arma::vec x = arma::linspace<arma::vec>(x_start, x_end, npoints);
    arma::vec y = arma::linspace<arma::vec>(y_start, y_end, npoints);

    arma::mat xx, yy;
    arma::meshgrid(x, y, xx, yy);
    arma::mat points = arma::join_horiz(arma::vectorise(xx), arma::vectorise(yy));

    int n = npoints * npoints;
    arma::mat simplex_indices;
    qh_new_qhull(2, n, points.memptr(), 0, "qhull d Qbb Qc Qz", &simplex_indices);

    arma::mat simplices = simplex_indices.rows(1, simplex_indices.n_rows - 1);
    arma::mat coordinates = points.rows(simplices.col(0));

    return std::make_pair(simplices, coordinates);
}

arma::mat Numerical_integration::Global_coordinates(int i)
{
    std::pair<arma::mat, arma::mat> res = construct_Delaunay();
    arma::mat simplices = res.first;
    arma::mat coordinates = res.second;
    int triangles_total = simplices.n_rows;
    if (i < triangles_total)
    {
        return coordinates.row(i);
    }
    else
    {
        return arma::mat
