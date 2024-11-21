#define _USE_MATH_DEFINES
//#define _ITERATOR_DEBUG_LEVEL 2 //not necessary
//#include <crtdbg.h> //not necessary
#include <iostream>


#include <iomanip> //for std::setprecision
#include <stdexcept>     // for std::runtime_error     

//#include <sstream>     //not needed 
#include <fstream>
#include <chrono> //to measure execution time
#include <string>
#include <vector>
#include <cmath>
#include <algorithm> 
#include <Eigen/Dense>       

using namespace std;

using namespace Eigen;




class SetGrid {
public: 
    size_t size;
    vector<double> u; //grid points from 0 to 1
    vector<double>  z; //chebyshev points from 1 to -1
    vector<vector<double>> D;  //chebyshev_diff_matrix
    vector<vector<double>> D2;
    
    SetGrid(size_t n) : size(n), z(n), u(n), D(n, vector<double>(n)){
        
        //set u_grid
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            z[i] = cos((M_PI * i) / (n - 1));
        }

        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            u[i] = z[i] * 0.5 + 0.5;
        }
    
        std::reverse(u.begin(), u.end());
    

        //chebyshev_diff_matrix
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
                if (i == static_cast<size_t>(n - 1) && j == static_cast<size_t>(n - 1)) {
                    D[i][j] = -((2.0 * (n - 1.0) * (n - 1) + 1.0) / 6.0);
                }
                else if (i == 0 && j == 0) {
                    D[i][j] = (2.0 * (n - 1.0) * (n - 1) + 1.0) / 6.0;
                }
                else if (i == j) {
                    D[i][j] = -z[i] / (2.0 * (1.0 - z[i] * z[i]));
                }
                else {
                    D[i][j] = (((i + j) % 2 == 0 ? 1.0 : -1.0) / (z[i] - z[j]));

                    

                    if (i == 0 || i == n - 1) D[i][j] *= 2.0;
                    if (j == 0 || j == n - 1) D[i][j] *= 1.0 / 2.0;
                }
            }
        }


        // Scale the differentiation matrix for the interval [0, 1]
        
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
                D[i][j] *= 2;
            }
        }

        // Reverse rows and columns to match increasing order in [0, 1]
        vector<vector<double>> Dtemp = D;

        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
                D[i][j] = Dtemp[static_cast<size_t>(n - 1 - i)][static_cast<size_t>(n - 1 - j)];
            }
        }
    
        D2 = multiplyMatrixMatrix(D, D);


    }

          

private:
        
        //function to compute the product of 2 matrices
        vector<vector<double>> multiplyMatrixMatrix(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
            size_t rows1 = mat1.size();
            size_t cols1 = mat1[0].size();
            size_t cols2 = mat2[0].size();

            vector<vector<double>> product(rows1, vector<double>(cols2, 0.0));

            for (size_t i = 0; i < rows1; ++i) {
                for (size_t j = 0; j < cols2; ++j) {
                    for (size_t k = 0; k < cols1; ++k) {
                        product[i][j] += mat1[i][k] * mat2[k][j];
                    }
                }
            }
            return product;
        }
};

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
class initialize {
public:
    initialize(SetGrid& grid, vector<vector<vector<double>>>& vals, double Beta, double Omega, double U0, double& ksi) :grid_inst(grid),
        beta(Beta), omega(Omega), u0(U0) {
        update(vals, ksi);
    }

    void update(vector<vector<vector<double>>>& vals, double& ksi) {
        vals[0][0][0] = Bs0(0);
        for (size_t i = 1; i < grid_inst.size; ++i) {
            vals[0][0][i] = aux(ksi, grid_inst.u[i]);
        }

        vals[0][1] = multiplyMatrixVector(grid_inst.D, vals[0][0]);
        vals[0][2] = multiplyMatrixVector(grid_inst.D2, vals[0][0]);
    }

private:
    double beta;
    double omega;
    double u0;
    SetGrid& grid_inst;


    //define the function Bs at the first time slice
    double Bs0(double uu) {
        return beta * exp(-pow((uu - u0), 2) / (2.0 * pow(omega, 2)));
    }


    double aux(const double& ksi, double uu) {
        return Bs0(uu / (1.0 + ksi * uu));
    }


    vector<double> multiplyMatrixVector(const vector<vector<double>>& matrix, const vector<double>& vec) {

        vector<double> result(matrix.size(), 0.0);

        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += matrix[i][j] * vec[j];
            }
        }
        return result;
    }


};

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

class coefficients {
public:
    vector<vector<double>> coeff_matrix;
    vector<double> source;

    coefficients(SetGrid& grid) : N(grid.size), coeff_matrix(grid.size, vector<double>(grid.size, 0.0)), source(grid.size, 0.0), 
        unit_matrix(grid.size, vector<double>(grid.size, 0.0)) {

        for (size_t i = 0; i < N; i++) {
            unit_matrix[i][i] = 1.0;
        }
    }
    
    //update the coeffs of the eom number m. m goes from 1 to 4.
    void update(size_t& m, SetGrid& grid_inst, vector<vector<vector<double>>>& data, double& ksi, double& a4){

        switch (m) {
        case 1: // Ss
            coeff_matrix[0][0] = 40.0;

            for (size_t i = 1; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    coeff_matrix[i][j] = 2.0 * pow(grid_inst.u[i], 2) * grid_inst.D2[i][j] + 20.0 * grid_inst.u[i] * 
                        grid_inst.D[i][j] + (40.0 + pow(grid_inst.u[i], 8) * pow(4 * data[0][0][i] + grid_inst.u[i] * 
                            data[0][1][i], 2)) * unit_matrix[i][j];
                }
            }

            source[0] = 0;
            for (size_t i = 1; i < N; ++i) {
                source[i] = -pow(grid_inst.u[i], 3) * (1.0 + grid_inst.u[i] * ksi) * pow(4 * data[0][0][i] + grid_inst.u[i] * 
                    data[0][1][i], 2);
            }
            break;
        case 2: // Sdots

            source[0] = 0.5 * a4;//bdry condition

            for (size_t i = 1; i < N; ++i) {
                source[i] = 2 * pow(grid_inst.u[i], 5) * pow(data[1][0][i], 2) + data[1][0][i] * (7.0 + 11.0 * grid_inst.u[i] * ksi + 4 *
                    pow(grid_inst.u[i], 2) * pow(ksi, 2)) + grid_inst.u[i] * pow(1 + grid_inst.u[i] * ksi, 2) * data[1][1][i];
            }

            // Create the coefficient matrix 
            coeff_matrix[0][0] = 1.0;

            for (size_t i = 1; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    coeff_matrix[i][j] = (-pow(grid_inst.u[i], 5) * data[1][0][i] - (1.0 + grid_inst.u[i] * ksi)) * grid_inst.D[i][j] -
                        2.0 * (ksi + 5.0 * pow(grid_inst.u[i], 4) * data[1][0][i] + pow(grid_inst.u[i], 5) * data[1][1][i]) * 
                        unit_matrix[i][j];
                }
            }
            break;
        case 3: // Bdots
            source[0] = 6.0 * data[0][0][0];
            for (size_t i = 1; i < N; ++i) {
                source[i] = 6.0 * data[0][0][i] * (pow(1.0 + grid_inst.u[i] * ksi, 2) + 2.0 * pow(grid_inst.u[i], 4) * data[2][0][i] ) +
                    0.5 * 3.0 * grid_inst.u[i] * (pow(1.0 + grid_inst.u[i] * ksi, 2) + 2.0 * pow(grid_inst.u[i], 4) * data[2][0][i]) * 
                    data[0][1][i];
            }

            // Create the coefficient matrix
           
            coeff_matrix[0][0] = -3.0;

            for (size_t i = 1; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    coeff_matrix[i][j] = -2.0 * grid_inst.u[i] * (1.0 + grid_inst.u[i] * ksi + pow(grid_inst.u[i], 5) * data[1][0][i]) *
                        grid_inst.D[i][j] - 3.0 * (1.0 + 2.0 * grid_inst.u[i] * ksi + 6.0 * pow(grid_inst.u[i], 5) * data[1][0][i] +
                            pow(grid_inst.u[i], 6) * data[1][1][i]) * unit_matrix[i][j];
                }
            }
            break;
        case 4: // As

            source[0] = a4;//bdry condition

            for (size_t i = 1; i < N; ++i) {
                source[i] = 3.0 * pow(grid_inst.u[i], 4) * data[3][0][i] * (4.0 * data[0][0][i] + grid_inst.u[i] * data[0][1][i]) *
                    pow(1.0 + grid_inst.u[i] * ksi + pow(grid_inst.u[i], 5) * data[1][0][i] , 2) -(6.0 / pow(grid_inst.u[i], 4)) * 
                    pow(1.0 + grid_inst.u[i] * ksi + pow(grid_inst.u[i], 5) * data[1][0][i] , 2) - (6.0 / pow(grid_inst.u[i], 4)) * 
                    (pow(1.0 + grid_inst.u[i] * ksi, 2) + 2.0 * pow(grid_inst.u[i], 4) * data[2][0][i] ) * (-1.0 + 4.0 * 
                        pow(grid_inst.u[i], 5) * data[1][0][i]  + pow(grid_inst.u[i], 6) * data[1][1][i]);
            }

            // Create the coefficient matrix 
            coeff_matrix[0][0] = 1.0;

            for (size_t i = 1; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    coeff_matrix[i][j] = pow(grid_inst.u[i], 2) * pow(1.0 + grid_inst.u[i] * ksi + pow(grid_inst.u[i], 5) *
                        data[1][0][i], 2) * grid_inst.D2[i][j] + 6.0 * grid_inst.u[i] * pow(1.0 + grid_inst.u[i] * ksi + 
                            pow(grid_inst.u[i], 5) * data[1][0][i] , 2) * grid_inst.D[i][j] + 6.0 * pow(1.0 + grid_inst.u[i] * ksi +
                                pow(grid_inst.u[i], 5) * data[1][0][i] , 2) * unit_matrix[i][j];
                }
            }

            break;
        }

    
    }
private:
    size_t N;
    vector<vector<double>> unit_matrix;
    
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

class solve{

public:
    coefficients coeff;

    solve(SetGrid& grid) : grid_inst(grid), coeff(grid) {}

    //solve the first num equations.
        void run(size_t num, vector<vector<vector<double>>>& data, double& ksi, double& a4) {
            for (size_t i = 1; i <= num; ++i) {
                coeff.update(i, grid_inst, data, ksi, a4);
                  

            // conversion to Eigen
            Eigen_coeff_matrix = convert_stdMatrix_To_MatrixXd(coeff.coeff_matrix);
            Eigen_source = Eigen::Map<Eigen::VectorXd>(coeff.source.data(), coeff.source.size());

            // solve
            Eigen::FullPivLU<Eigen::MatrixXd> lu(Eigen_coeff_matrix);
            Eigen_sol = lu.solve(Eigen_source);

            // Convert vectorXd to vector
            data[i][0] = convert_vactorXd_to_stdVector(Eigen_sol);
            data[i][1] = multiplyMatrixVector(grid_inst.D, data[i][0]);
            data[i][2] = multiplyMatrixVector(grid_inst.D2, data[i][0]);
            
        }
    
        }

    

private: 
    SetGrid& grid_inst;
    Eigen::MatrixXd Eigen_coeff_matrix;
    Eigen::VectorXd Eigen_source;
    Eigen::VectorXd Eigen_sol;
    

    vector<double> multiplyMatrixVector(const vector<vector<double>>& matrix, const vector<double>& vec) {

        vector<double> result(matrix.size(), 0.0);

        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += matrix[i][j] * vec[j];
            }
        }
        return result;
    }



    // Function to convert Matrix to a matrixXd
    Eigen::MatrixXd convert_stdMatrix_To_MatrixXd(const vector<vector<double>>& matrix_xd) {
        size_t rows = matrix_xd.size();
        size_t cols = matrix_xd[0].size();
        Eigen::MatrixXd mat(rows, cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat(i, j) = matrix_xd[i][j];
            }
        }
        return mat;
    }

    // Function to convert vectorXd to vector
    vector<double> convert_vactorXd_to_stdVector(const Eigen::VectorXd& vec) {
        return std::vector<double>(vec.data(), vec.data() + vec.size());
    }


};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

class interpolate {
public:
    interpolate(SetGrid& grid) : grid_inst(grid), N(grid.size) {}

    //chebyshev interpolation: interpolation of a function (whose values on the grid points from 0 to 1 are known) using chebyshev expansion
    double cheby(const vector<double>& f, double y) {
        double var = 0.0;
        for (size_t j = 1; j <= N; ++j) {
            var += cardinal_fct(j, y) * f[static_cast<size_t>(N - j)];
        }
        return var;

    }


    // interpolation function
    double polynomial(const std::vector<double>& x, const std::vector<double>& y, double x_val, int order) {
        switch (order) {
        case 0: // Nearest neighbor
            return nearest_neighbor_interpolation(x, y, x_val);
        case 1: // Linear interpolation 
            return linear_interpolation(x, y, x_val);
        case 2: // Quadratic
        case 3: // Cubic
        case 4: // Quartic
            return polynomial_interpolation(x, y, x_val, order);

        }
    }


    double trapezoidal_integral(const std::vector<double>& x, const std::vector<double>& y, double start, double end, int order,
        size_t n_steps) {
        double step_size = (end - start) / n_steps;
        double integral = 0.0;


        for (size_t i = 0; i < n_steps; ++i) {
            double x_left = start + i * step_size;
            double x_right = start + (i + 1) * step_size;
            double y_left = polynomial(x, y, x_left, order);
            double y_right = polynomial(x, y, x_right, order);
            integral += 0.5 * (y_left + y_right) * step_size;
        }

        return integral;
    }

private:
    SetGrid& grid_inst;
    size_t N;

    //function to calculate the chebyshev polynomials of first kind, which are defined on [-1,1]
    //double cheby_polynomial_1_kind(int j, double y) {
    //    return cos(j * acos(y));
    //}

    //function to calculate the chebyshev polynomials of second kind, which are defined on [-1,1]

    double cheby_polynomial_2_kind(size_t j, double y) {
        if (j == 0) return 1;
        if (j == 1) return 2 * y;

        double U_n_minus_2 = 1;
        double U_n_minus_1 = 2 * y;
        double U_n = 0;

        for (size_t i = 2; i <= j; ++i) {
            U_n = 2 * y * U_n_minus_1 - U_n_minus_2;
            U_n_minus_2 = U_n_minus_1;
            U_n_minus_1 = U_n;
        }

        return U_n;
    }

    //function to calculate the cardinal functions for the chebyshev expansion for [-1,1]. 
    double cardinal_fct_oneToMinusOne(size_t j, double y) {//j is for the order of the cardinal fct
        vector<double> z = grid_inst.z;

        if (y == z[j - 1]) return 1.0;

        double aux = (j == 1 || j == N) ? 2.0 : 1.0;
        return pow(-1, j) * (1 - y * y) * (N - 1) * cheby_polynomial_2_kind(N - 2, y) / (aux * pow(N - 1, 2) * (y - z[j - 1]));
    }


    //function to calculate the  cardinal functions for the chebyshev expansion for [0,1].
    double cardinal_fct(size_t j, double y) {//j is for the order of the cardinal fct
        return cardinal_fct_oneToMinusOne(j, 2.0 * y - 1.0);
    }

    //for the time interpolation
    double nearest_neighbor_interpolation(const std::vector<double>& x, const std::vector<double>& y, double x_val) {
        auto it = std::lower_bound(x.begin(), x.end(), x_val);
        if (it == x.begin()) return y.front();
        if (it == x.end()) return y.back();
        auto index = std::distance(x.begin(), it);
        if (x_val - x[index - 1] < x[index] - x_val) {
            return y[index - 1];
        }
        else {
            return y[index];
        }
    }


    double linear_interpolation(const std::vector<double>& x, const std::vector<double>& y, double x_val) {

        if (x_val < x.front()) {
            double x0 = x[0], x1 = x[1];
            double y0 = y[0], y1 = y[1];
            return y0 + (y1 - y0) * (x_val - x0) / (x1 - x0);
        }

        if (x_val > x.back()) {
            size_t n = x.size();
            double x0 = x[n - 2], x1 = x[n - 1];
            double y0 = y[n - 2], y1 = y[n - 1];
            return y0 + (y1 - y0) * (x_val - x0) / (x1 - x0);
        }


        auto it = std::lower_bound(x.begin(), x.end(), x_val);
        auto index = std::distance(x.begin(), it);
        double x0 = x[index - 1], x1 = x[index];
        double y0 = y[index - 1], y1 = y[index];
        return y0 + (y1 - y0) * (x_val - x0) / (x1 - x0);
    }

    // Polynomial interpolation for orders 2, 3, and 4
    double polynomial_interpolation(const std::vector<double>& x, const std::vector<double>& y, double x_val, int order) {
        int n = static_cast<int>(x.size());

        int subset_size = std::min(order + 1, n);


        int start_idx = 0;
        auto it = std::lower_bound(x.begin(), x.end(), x_val);
        int index = static_cast<int>(std::distance(x.begin(), it));


        if (index >= subset_size / 2 && index + subset_size / 2 < n) {
            start_idx = index - subset_size / 2;
        }
        else if (index + subset_size / 2 >= n) {
            start_idx = n - subset_size;
        }

        std::vector<double> x_sub(x.begin() + start_idx, x.begin() + start_idx + subset_size);
        std::vector<double> y_sub(y.begin() + start_idx, y.begin() + start_idx + subset_size);


        double result = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(subset_size); ++i) {
            double term = y_sub[i];
            for (size_t j = 0; j < static_cast<size_t>(subset_size); ++j) {
                if (i != j) {
                    term *= (x_val - x_sub[j]) / (x_sub[i] - x_sub[j]);
                }
            }
            result += term;
        }
        return result;
    }


};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

class find_horizon {
public:

    double run(vector<vector<vector<double>>>& vals, const double& ksi, double& initial_guess, double& tolerance, int& max_iterations) {
        double x_n = initial_guess;

        for (size_t i = 0; i < static_cast<size_t>(max_iterations); ++i) {
            double f_xn = (1.0 / (2.0 * pow(x_n, 2))) + pow(x_n, 2) * interp.cheby(vals[2][0], x_n) + (ksi / x_n) +
                0.5 * pow(ksi, 2);
            double df_xn = -(1.0 / pow(x_n, 3)) + 2.0 * x_n * interp.cheby(vals[2][0], x_n) - (ksi / pow(x_n, 2)) +
                pow(x_n, 2) * interp.cheby(vals[2][1], x_n);

            if (abs(f_xn) < tolerance) {return x_n;}
            x_n = x_n - (f_xn / df_xn);
        }
        std::cerr << "While looking for horizon position, max number of iterations is reached." << std::endl;//unnecessary
        return x_n;
    }


    find_horizon(SetGrid& grid) : interp(grid) {}


private:

    interpolate interp;

};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class radial_shift {

public:
    void run(double& uh, double desired_uh, double& ksi) {
        ksi += -(1.0 / ( desired_uh) ) + (1.0/uh);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Wilke_Routine {
private:
    vector<double> dts;
    int number_slices;
    int orderInterp;
    double tolerance ;
    int max_iterations ;
    double tt;
    double deltaP_over_Peq;
    double dt_ksi;
    vector<double> last_dtBs;

    interpolate interp;
    double change_in_ksi;//delete me
    

public:
    vector<double> horizonlist;
    vector<double> timelist;
    vector<double> timelist_Bs;
    vector<double> deltaP_over_Peq_list;

    vector<vector<double>> Ss_list;//contains Ss at all times
    vector<vector<double>> Sdots_list;
    vector<vector<double>> Bdots_list;
    vector<vector<double>> As_list;
    vector<vector<double>> Bs_list;
    vector<vector<double>> dtBs_list;//contains time derivative of Bs at all times.

    vector<double> ksi_list;
    vector<double> dt_ksi_list;//contains time derivative of ksi at all times.


    Wilke_Routine(SetGrid& grid, double&t_0, double& t_f, double& dt, vector<vector<vector<double>>>& vals, double& ksi, double& a4, 
        solve& solver, find_horizon& horizon_finder, double& uh, initialize& Initialize, radial_shift& shift): tt(t_0), interp(grid), 
        orderInterp(5),number_slices(static_cast<int>(std::ceil((t_f - t_0) / dt))), tolerance(1e-10), max_iterations(10000), 
        last_dtBs(grid.size), change_in_ksi(0.0){
        //number_slices(static_cast<int>(std::ceil((t_f - t_0) / dt))), tolerance(1e-7), max_iterations(2000), last_dtBs(grid.size) {
        
        //create table dts
        for (int i = -8; i <= -1; ++i) {
            dts.push_back(pow(10.0, i / 2.0));
        }

        double sum_first_part = 0.0;
        for (size_t i = 0; i <= 7; ++i) {
            sum_first_part += dts[i];
        }
        dts.push_back(1.0 - sum_first_part);
        dts.push_back(1.0);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        Initialize.update(vals, ksi);

        std::cout << "number of time slices is: " << number_slices << endl;

        for (size_t count = 1; count <= static_cast<size_t>(number_slices); ++count) {

            if (count % 100 == 0) {
                solver.run(2, vals, ksi, a4);
                double ksi_before_shift=ksi;
                uh = horizon_finder.run(vals, ksi, uh, tolerance, max_iterations);
                shift.run(uh, 0.8, ksi);
                vector<double> aux_vector(grid.size);

                for (size_t i = 0; i < grid.size; ++i) {
                    aux_vector[i]= interp.cheby ( vals[0][0], grid.u[i] / (1.0 + (ksi- ksi_before_shift)* grid.u[i]  ) );
                }
                for (size_t i = 0; i < grid.size; ++i) {
                    vals[0][0][i] = aux_vector[i];
                }
            }

            solver.run(4, vals, ksi, a4);
            
            uh = horizon_finder.run(vals, ksi, uh, tolerance, max_iterations);
        
            cout << "uh= " << uh << endl;//delete me
            horizonlist.push_back(uh);
            timelist.push_back(tt);

            Bs_list.push_back(vals[0][0]);
            Ss_list.push_back(vals[1][0]);
            Sdots_list.push_back(vals[2][0]);
            Bdots_list.push_back(vals[3][0]);
            As_list.push_back(vals[4][0]);
            

            deltaP_over_Peq = (12.0 / pow(M_PI, 2)) * vals[0][0][0] / a4;
            deltaP_over_Peq_list.push_back(deltaP_over_Peq);

            
            


            dt_ksi = (2.0 + 2.0 * pow(uh, 4) * interp.cheby(vals[4][0], uh) + pow(uh, 8) * pow(interp.cheby(vals[3][0], uh), 2) +
                4.0 * uh * ksi + 2.0 * pow(uh, 2) * pow(ksi, 2)) / (4.0 * pow(uh, 2)); // dt_ksi is time derivative of ksi.

            dt_ksi_list.push_back(dt_ksi);


            last_dtBs[0] = 0.5 * (8.0 * vals[0][0][0] * ksi + 2.0 * vals[3][1][0]  + 5.0 * vals[0][1][0] );
            for (size_t j = 1; j < grid.size; ++j) {
                last_dtBs[j] = (1.0 / (2.0 * grid.u[j])) * (2.0 * vals[3][0][j]  + (1.0 + pow(grid.u[j], 4) * vals[4][0][j]
                    + 2.0 * grid.u[j] * ksi + pow(grid.u[j], 2) * pow(ksi, 2) -
                    2.0 * pow(grid.u[j], 2) * dt_ksi) * (4.0 * vals[0][0][j]  + grid.u[j] * vals[0][1][j] ));
            }


            dtBs_list.push_back(last_dtBs);

            tt = tt + dt * dts[min(dts.size(), timelist.size()) - 1];

            if (static_cast<int>(timelist.size()) > 5 &&
                (timelist[timelist.size() - 6] - timelist[timelist.size() - 5]) -
                (timelist[timelist.size() - 3] - timelist[timelist.size() - 2]) < 1e-12) {


                //get Bs at next time slice
                for (size_t k = 0; k < grid.size; ++k) {
                    vals[0][0][k] += (dt / 720.0) * (1901.0 * dtBs_list[dtBs_list.size() - 1][k]
                        - 2774.0 * dtBs_list[dtBs_list.size() - 2][k]
                        + 2616.0 * dtBs_list[dtBs_list.size() - 3][k]
                        - 1274.0 * dtBs_list[dtBs_list.size() - 4][k]
                        + 251.0 * dtBs_list[dtBs_list.size() - 5][k]);
                }



                //get ksi at next time slice
                change_in_ksi= (dt / 720.0) * (1901.0 * dt_ksi_list[dt_ksi_list.size() - 1]
                    - 2774.0 * dt_ksi_list[dt_ksi_list.size() - 2]
                    + 2616.0 * dt_ksi_list[dt_ksi_list.size() - 3]//delete me
                    - 1274.0 * dt_ksi_list[dt_ksi_list.size() - 4]
                    + 251.0 * dt_ksi_list[dt_ksi_list.size() - 5]);

                    ksi += (dt / 720.0) * (1901.0 * dt_ksi_list[dt_ksi_list.size() - 1]
                        - 2774.0 * dt_ksi_list[dt_ksi_list.size() - 2]
                        + 2616.0 * dt_ksi_list[dt_ksi_list.size() - 3]
                        - 1274.0 * dt_ksi_list[dt_ksi_list.size() - 4]
                        + 251.0 * dt_ksi_list[dt_ksi_list.size() - 5]);
                
            }
            else {


                int InterpolationOrder_Bs_ksi = std::min(orderInterp - 1, static_cast<int>(dtBs_list.size()) - 1);

                //get Bs at next time slice

                for (size_t i = 0; i < grid.size; ++i) {

                    vector<double> rel_time_list;
                    vector<double> function_values;
                    size_t mm = static_cast<size_t>(min(static_cast<int>(dtBs_list.size()), 6));

                    for (size_t j = 1; j <= mm; ++j) {
                        rel_time_list.push_back(timelist[timelist.size() - (mm - j + 1)]);
                        function_values.push_back(dtBs_list[timelist.size() - (mm - j + 1)][i]);
                    }




                    vals[0][0][i] += interp.trapezoidal_integral(rel_time_list, function_values, timelist[timelist.size() - 1], tt, InterpolationOrder_Bs_ksi, 5000);




                }


                //get ksi at next time slice

                vector<double> rel_time_list;
                vector<double> function_values_dtksi;
                size_t mm = static_cast<size_t>(min(static_cast<int>(dt_ksi_list.size()), 6));

                for (size_t j = 1; j <= mm; ++j) {
                    rel_time_list.push_back(timelist[timelist.size() - (mm - j + 1)]);
                    function_values_dtksi.push_back(dt_ksi_list[timelist.size() - (mm - j + 1)]);
                }


                change_in_ksi= interp.trapezoidal_integral(rel_time_list, function_values_dtksi, timelist[timelist.size() - 1],
                    tt, InterpolationOrder_Bs_ksi, 10000);

                ksi += interp.trapezoidal_integral(rel_time_list, function_values_dtksi, timelist[timelist.size() - 1], 
                    tt, InterpolationOrder_Bs_ksi, 10000);//tunable parameter


            }

            vector<double> aux_vector(grid.size);
            for (size_t i = 0; i < grid.size; ++i) {
                   aux_vector[i]= interp.cheby ( vals[0][0], grid.u[i] / (1.0 +  change_in_ksi * grid.u[i]  ) );
               }
               for (size_t i = 0; i < grid.size; ++i) {
                   vals[0][0][i] = aux_vector[i];
               }



            
            vals[0][1] = multiplyMatrixVector(grid.D, vals[0][0]); 
            vals[0][2] = multiplyMatrixVector(grid.D2, vals[0][0]);


            ksi_list.push_back(ksi);
            cout << "time slice " << count << endl;
        }


        for (size_t i = 0; i < timelist.size(); ++i) {
            timelist_Bs.push_back(timelist[i]);
        }

        timelist_Bs.push_back(tt);

        Bs_list.push_back(vals[0][0]);



        double deltaP_over_Peq = (12.0 / pow(M_PI, 2)) * vals[0][0][0] / a4;
        deltaP_over_Peq_list.push_back(deltaP_over_Peq);

    }
    

private:
    vector<double> multiplyMatrixVector(const vector<vector<double>>& matrix, const vector<double>& vec) {

        vector<double> result(matrix.size(), 0.0);

        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += matrix[i][j] * vec[j];
            }
        }
        return result;
    }



};

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
class print {

public:
    print(vector<vector<vector<double>>>& vals, Wilke_Routine& num_routine, SetGrid& grid, int Precision)
        : precision(Precision) {

        writeDeltaPOverPeq(num_routine);
        writeMatrixToFile(num_routine.Bs_list, "Bs.dat", num_routine.timelist_Bs, grid.u);
        writeMatrixToFile(num_routine.Ss_list, "Ss.dat", num_routine.timelist, grid.u);
        writeMatrixToFile(num_routine.Sdots_list, "Sdots.dat", num_routine.timelist, grid.u);
        writeMatrixToFile(num_routine.Bdots_list, "Bdots.dat", num_routine.timelist, grid.u);
        writeMatrixToFile(num_routine.As_list, "As.dat", num_routine.timelist, grid.u);
    }

private:
    int precision;

    void writeDeltaPOverPeq(Wilke_Routine& num_routine) {
        std::ofstream outFile("deltaP_over_Peq.dat");
        if (!outFile) {
            throw std::runtime_error("Unable to open deltaP_over_Peq.dat for writing!");
        }
        //set precision
        outFile << std::fixed << std::setprecision(precision);
        
        for (size_t i = 0; i < num_routine.timelist_Bs.size(); ++i) {
            outFile << num_routine.timelist_Bs[i] << " " << num_routine.deltaP_over_Peq_list[i] << endl;
        }
    }

    void writeMatrixToFile(vector<vector<double>>& matrix, string filename,
        vector<double>& t_values, vector<double>& u_values) {
        std::ofstream ofs(filename); 
        if (!ofs) {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        ofs << std::fixed << std::setprecision(precision);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                ofs << t_values[i] << " " << u_values[j] << " " << matrix[i][j] << "\n";
            }
            ofs << "\n"; 
        }
    }
};




int main() {

    //_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
    auto start = std::chrono::high_resolution_clock::now();

    double ksi=0.0;
    

    SetGrid grid(40); // with number of collocation points
    vector<vector<vector<double>>> vals(5, vector<vector<double>>(3, vector<double>(grid.size)));

    double a4 = -3.0; //initial energy density. Is constant over time.

    /////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    // Bs at first time slice is defined as beta * exp(-pow((uu - u0), 2) / (2.0 * pow(omega, 2))). 
    // Choose now values of beta, omega then u0.  
    initialize Initialize(grid, vals, 0.9, 0.05, 0.5, ksi);


    solve solver(grid);   

    // Run the solver
    solver.run(2, vals, ksi, a4);

    //Now find horizon
    double uh = 1.0;  
    double tolerance = 1e-10; 
    int max_iterations = 10000;

    find_horizon horizon_finder(grid);
    uh = horizon_finder.run(vals, ksi, uh, tolerance, max_iterations);
    
    //shift horizon to u=1.0
    radial_shift shift;
    shift.run(uh, 0.8, ksi);
    

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double t_0 = 0.0; // Initial time
    double dt = 1.0 / 2000.0; // Time step
    double t_f = 3.0; // Final time //change back to 3.0

    
    Wilke_Routine  num_routine(grid, t_0, t_f, dt, vals, ksi, a4, solver, horizon_finder, uh, Initialize, shift);

    auto end_numerics = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> duration_numerics = end_numerics - start;

    

   // std::cout << "Ss(u) at the last time slice:" << endl;
    //for (size_t i = 0; i < grid.size; ++i) {
     //   std::cout << "Ss(" << grid.u[i] << ") = " << vals[1][0][i] << endl;
    //}
    //std::cout << endl;


    //std::cout << "Sdots(u) at the last time slice:" << endl;
    //for (size_t i = 0; i < grid.u.size(); ++i) {
    //    std::cout << "Sdots(" << grid.u[i] << ") = " << vals[2][0][i] << endl;
   // }
    //std::cout << endl;

    //std::cout << "Bdots(u) at the last time slice:" << endl;
    //for (size_t i = 0; i < grid.u.size(); ++i) {
    //    std::cout << "Bdots(" << grid.u[i] << ") = " << vals[3][0][i] << endl;
    //}
    //std::cout << endl;


    //std::cout << "As(u) at the last time slice:" << endl;
    //for (size_t i = 0; i < grid.u.size(); ++i) {
    //    std::cout << "As(" << grid.u[i] << ") = " << vals[4][0][i] << endl;
    //}
    //std::cout << endl;


    //std::cout << "Bs(u) at the last time slice:" << endl;
    //for (size_t i = 0; i < grid.u.size(); ++i) {
    //    std::cout << "Bs(" << grid.u[i] << ") = " << vals[0][0][i] << endl;
    //}
    //std::cout << endl;

    
    
    //Print out results in seperate dat files
    print Print(vals, num_routine, grid, 17);


    // Calculate the program execution time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Numerical computations time, before printing out result to dat files: " << duration_numerics.count() << " seconds" << std::endl;
    std::cout << "Program execution time: " << duration.count() << " seconds" << std::endl;
    


    //print horiozn list
    std::ofstream outFile("horizon list.dat");
    if (!outFile) {
        throw std::runtime_error("Unable to open horizon list.dat for writing!");
    }
    //set precision
    outFile << std::fixed << std::setprecision(17);

    for (size_t i = 0; i < num_routine.timelist.size(); ++i) {
        outFile << num_routine.timelist[i] << " " << num_routine.horizonlist[i] << endl;
    }
    

    return 0;
}