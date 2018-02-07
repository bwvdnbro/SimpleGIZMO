#include "RiemannSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

/**
 * @brief Cubic spline kernel (without smoothing length normalization).
 *
 * @param u Normalized coordinate (in units of the smoothing length).
 * @param W Resulting value of the kernel.
 */
void cubic_spline_kernel(const double u, double &W) {
  // expression taken from Matthieu's SWIFT paper draft, and Dehnen & Aly (2012)
  if (u < 1.) {
    const double u2 = u * u;
    const double u3 = u * u2;
    if (u < 0.5) {
      W = 3. * (u3 - u2) + 0.5;
    } else {
      W = -u3 + 3. * (u2 - u) + 1.;
    }
  } else {
    // the cubic spline kernel has compact support
    W = 0.;
  }
  W *= (8. / 3.);
}

/**
 * @brief Cubic spline kernel and first derivative (without smoothing length
 * normalization).
 *
 * @param u Normalized coordinate (in units of the smoothing length).
 * @param W Resulting value of the kernel.
 * @param dW_du Resulting first derivative of the kernel.
 */
void cubic_spline_kernel(const double u, double &W, double &dW_du) {
  // expression taken from Matthieu's SWIFT paper draft, and Dehnen & Aly (2012)
  if (u < 1.) {
    const double u2 = u * u;
    const double u3 = u * u2;
    if (u < 0.5) {
      W = 3. * (u3 - u2) + 0.5;
      dW_du = 9. * u2 - 6. * u;
    } else {
      W = -u3 + 3. * (u2 - u) + 1.;
      dW_du = -3. * u2 + 6. * u - 3.;
    }
  } else {
    // the cubic spline kernel has compact support
    W = 0.;
    dW_du = 0.;
  }
  W *= (8. / 3.);
  dW_du *= (8. / 3.);
}

/**
 * @brief Particle data.
 */
class Particle {
public:
  /// Geometrical quantities.

  /*! @brief Particle position. */
  double _position;
  /*! @brief Particle velocity. */
  double _particle_velocity;
  /*! @brief Smoothing length. */
  double _smoothing_length;
  /*! @brief Derivative of the kernel sum w.r.t. the smoothing length. */
  double _wcount_dh;
  /*! @brief Kernel sum/mesh-free volume. */
  double _volume;
  /*! @brief Gradient matrix. */
  double _matrix;

  /// Conserved quantities.

  /*! @brief Fluid mass. */
  double _mass;
  /*! @brief Fluid momentum. */
  double _momentum;
  /*! @brief Fluid total energy. */
  double _energy;

  /// Primitive quantities.

  /*! @brief Fluid density. */
  double _density;
  /*! @brief Fluid velocity. */
  double _fluid_velocity;
  /*! @brief Fluid pressure. */
  double _pressure;

  /**
   * @brief Empty constructor.
   */
  Particle()
      : _position(0.), _particle_velocity(0.), _smoothing_length(0.),
        _volume(0.), _matrix(0.), _mass(0.), _momentum(0.), _energy(0.),
        _density(0.), _fluid_velocity(0.), _pressure(0.) {}
};

/**
 * @brief Periodic shortest (signed) distance between two particles.
 *
 * @param p1 First particle.
 * @param p2 Second particle.
 * @return First particle position minus second particle position, corrected for
 * periodic boundary jumps.
 */
double distance(const Particle &p1, const Particle &p2) {
  double d = p1._position - p2._position;
  if (d > 0.5) {
    d -= 1.;
  }
  if (d < -0.5) {
    d += 1.;
  }
  return d;
}

/**
 * @brief Dump the particle data to the given stream for output.
 *
 * @param particles Particles.
 * @param stream std::ostream to write to (can be std::cout for output to the
 * terminal, or a std::ofstream for file output).
 */
void dump(const std::vector<Particle> &particles, std::ostream &stream) {
  for (size_t i = 0; i < particles.size(); ++i) {
    const Particle &particle = particles[i];
    stream << particle._position << "\t" << particle._density << "\t"
           << particle._fluid_velocity << "\t" << particle._pressure << "\n";
  }
}

/**
 * @brief Solve for the flux between the given left and right state, using the
 * given Riemann solver.
 *
 * Note that we always assume a coordinate frame in which the left state is
 * spatially located to the left of the right state! We therefore need to
 * provide the actual surface normal to the interface and transform input and
 * output values to and from this coordinate frame.
 *
 * @param solver RiemannSolver to use.
 * @param gamma Polytropic index of the fluid.
 * @param rhoL Left state fluid density.
 * @param uL Left state fluid velocity.
 * @param PL Left state fluid pressure.
 * @param rhoR Right state fluid density.
 * @param uR Right state fluid velocity.
 * @param PR Right state fluid pressure.
 * @param n_unit Unit vector of the actual interface between left and right
 * state (used for coordinate transformation to the solver frame).
 * @param vframe Interface velocity (used to deboost).
 * @param flux Resulting fluxes.
 */
void riemann_solve_for_flux(const RiemannSolver &solver, const double gamma,
                            const double rhoL, const double uL, const double PL,
                            const double rhoR, const double uR, const double PR,
                            const double n_unit, double vframe,
                            double flux[3]) {

  // transform to the solver coordinate frame
  const double vL = (uL - vframe) * n_unit;
  const double vR = (uR - vframe) * n_unit;

  // solve the Riemann problem
  double rhosol, vsol, Psol;
  solver.solve(rhoL, vL, PL, rhoR, vR, PR, rhosol, vsol, Psol);

  // transform back to the original coordinate frame
  vsol *= n_unit;

  // compute the fluxes
  const double vtot = vsol + vframe;
  flux[0] = rhosol * vsol;
  flux[1] = rhosol * vtot * vsol + Psol;
  flux[2] =
      (Psol / (gamma - 1.) + 0.5 * rhosol * vtot * vtot) * vsol + Psol * vtot;
}

/**
 * @brief Do the density update for the given pair of particles.
 *
 * @param pi Active particle, is updated.
 * @param pj Neighbouring particle, is not updated.
 * @param d Relative distance between the two particles (the result of
 * distance(pi, pj).
 */
void do_density(Particle &pi, const Particle &pj, const double d) {

  // evaluate the kernel and derivative
  const double u = std::abs(d) / pi._smoothing_length;
  double W, dW_du;
  cubic_spline_kernel(u, W, dW_du);

  // update the active particle counters
  pi._volume += W;
  pi._wcount_dh -= (W + u * dW_du);
  pi._matrix += d * d * W;
}

/**
 * @brief Do the flux exchange for the given pair of particles.
 *
 * @param pi Active particle, is updated.
 * @param pj Neighbouring particle, is not updated.
 * @param d Relative distance between the two particles (the result of
 * distance(pi, pj).
 * @param solver RiemannSolver to use for the flux exchange.
 * @param gamma Polytropic index of the fluid.
 * @param dt Time step with which to exchange the flux.
 */
void do_flux(Particle &pi, const Particle &pj, const double d,
             const RiemannSolver &solver, const double gamma, const double dt) {

  // evaluate the kernels
  const double hi = pi._smoothing_length;
  const double hj = pj._smoothing_length;
  const double ui = std::abs(d) / hi;
  const double uj = std::abs(d) / hj;
  double Wi, Wj;
  cubic_spline_kernel(ui, Wi);
  cubic_spline_kernel(uj, Wj);

  // compute the interface oriented surface area and normal
  const double A = -pi._volume * pi._matrix * d * Wi / hi -
                   pj._volume * pj._matrix * d * Wj / hj;
  const double n_unit = A / std::abs(A);

  // compute the interface velocity
  const double xfac = hi / (hi + hj);
  const double vface = pi._particle_velocity -
                       (pi._particle_velocity - pj._particle_velocity) * xfac;

  // compute the flux
  double flux[3];
  riemann_solve_for_flux(solver, gamma, pi._density, pi._fluid_velocity,
                         pi._pressure, pj._density, pj._fluid_velocity,
                         pj._pressure, n_unit, vface, flux);

  // update the conserved variables with the flux contributions
  pi._mass -= flux[0] * A * dt;
  pi._momentum -= flux[1] * A * dt;
  pi._energy -= flux[2] * A * dt;
}

/**
 * @brief Very basic 1D GIZMO implementation to test the algorithm.
 *
 * @param argc Number of command line arguments (ignored).
 * @param argv Command line arguments (ignored).
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  /// global parameters

  // hydro related
  const double gamma = 5. / 3.;
  const RiemannSolver solver(gamma);

  // particle related
  const unsigned int number_of_particles = 100;

  // time integration related
  const double dt = 0.001;
  const unsigned int number_of_steps = 100;

  /// derived parameters

  // initial inter-particle distance
  const double particle_size = 1. / number_of_particles;

  /// setup

  // set up the particle and sort array for fast neighbour location
  std::vector<Particle> particles(number_of_particles);
  std::vector<unsigned int> sort_order(particles.size(), 0);
  for (size_t i = 0; i < particles.size(); ++i) {
    sort_order[i] = i;
  }

  // set up the initial condition
  // we immediately initialize the conserved variables, assuming a perfect
  // particle volume
  for (size_t i = 0; i < particles.size(); ++i) {
    particles[i]._position = (i + 0.5) * particle_size;
    particles[i]._smoothing_length = 2. * particle_size;
    if (particles[i]._position < 0.5) {
      particles[i]._mass = 1. * particle_size;
      particles[i]._energy = 1. * particle_size / (gamma - 1.);
    } else {
      particles[i]._mass = 0.125 * particle_size;
      particles[i]._energy = 0.1 * particle_size / (gamma - 1.);
    }
  }

  /// simulation

  // time integration loop
  for (unsigned int iloop = 0; iloop < number_of_steps; ++iloop) {

    // sort the particles for quick 1D neigbour finding
    std::sort(sort_order.begin(), sort_order.end(),
              [&particles](size_t i1, size_t i2) {
                return particles[i1]._position < particles[i2]._position;
              });

    // initialize volume quantities
    for (size_t i = 0; i < particles.size(); ++i) {
      Particle &particle = particles[i];
      particle._volume = 0.;
      particle._wcount_dh = 0.;
      particle._matrix = 0.;
    }

    // volume computation
    double totvol = 0.;
    for (size_t i = 0; i < sort_order.size(); ++i) {
      Particle &particle = particles[sort_order[i]];
      // we redo this loop until we find a converged smoothing length
      while (particle._volume == 0.) {
        // first do the particles with lower indices
        size_t ilow = (i > 0) ? i - 1 : sort_order.size() - 1;
        double d = distance(particle, particles[sort_order[ilow]]);
        while (std::abs(d) <= particle._smoothing_length) {
          do_density(particle, particles[sort_order[ilow]], d);
          // update index
          ilow = (ilow > 0) ? ilow - 1 : sort_order.size() - 1;
          d = distance(particle, particles[sort_order[ilow]]);
        }
        // now do the particles with higher indices
        size_t ihigh = (i < sort_order.size() - 1) ? i + 1 : 0;
        d = distance(particle, particles[sort_order[ihigh]]);
        while (std::abs(d) <= particle._smoothing_length) {
          do_density(particle, particles[sort_order[ihigh]], d);
          // update index
          ihigh = (ihigh < sort_order.size() - 1) ? ihigh + 1 : 0;
          d = distance(particle, particles[sort_order[ihigh]]);
        }
        // add the particle self-contribution
        do_density(particle, particle, 0.);

        // normalize quantities
        particle._volume /= (particle._smoothing_length);
        particle._wcount_dh /=
            (particle._smoothing_length * particle._smoothing_length);
        particle._matrix /= particle._smoothing_length;

        // the factor 1.732051 relates the actual support of the kernel to the
        // true smoothing scale (see Dehnen & Aly, 2012)
        const double n_sum =
            particle._volume * particle._smoothing_length / 1.732051;
        const double n_target = 1.235;
        const double f = n_sum - n_target;
        const double f_prime =
            particle._wcount_dh * particle._smoothing_length / 1.732051 +
            particle._volume;
        double h_new = particle._smoothing_length - f / f_prime;
        h_new = std::min(h_new, 2. * particle._smoothing_length);
        h_new = std::max(h_new, 0.5 * particle._smoothing_length);
        if (std::abs(h_new - particle._smoothing_length) >
            1.e-4 * particle._smoothing_length) {
          // redo this particle
          particle._smoothing_length = h_new;
          particle._volume = 0.;
          particle._wcount_dh = 0.;
          particle._matrix = 0.;
        }
      }

      // now get the volume
      particle._volume = 1. / particle._volume;
      particle._matrix = 1. / particle._matrix;
      totvol += particle._volume;

      // compute primitive variables
      particle._density = particle._mass / particle._volume;
      particle._fluid_velocity = particle._momentum / particle._mass;
      particle._pressure =
          (gamma - 1.) * (particle._energy -
                          0.5 * particle._fluid_velocity * particle._momentum) /
          particle._volume;
    }

    // set the particle velocities
    for (size_t i = 0; i < particles.size(); ++i) {
      Particle &particle = particles[i];
      particle._particle_velocity = particle._fluid_velocity;
    }

    // sanity check the total volume
    if (std::abs(totvol - 1.) > 0.01) {
      std::cerr << "Volume error!" << std::endl;
    }

    // flux exchange
    for (size_t i = 0; i < sort_order.size(); ++i) {
      Particle &particle = particles[sort_order[i]];
      // first do the particles with lower indices
      size_t ilow = (i > 0) ? i - 1 : sort_order.size() - 1;
      double d = distance(particle, particles[sort_order[ilow]]);
      while (std::abs(d) <= particle._smoothing_length) {
        do_flux(particle, particles[sort_order[ilow]], d, solver, gamma, dt);
        // update index
        ilow = (ilow > 0) ? ilow - 1 : sort_order.size() - 1;
        d = distance(particle, particles[sort_order[ilow]]);
      }
      // now do the particles with higher indices
      size_t ihigh = (i < sort_order.size() - 1) ? i + 1 : 0;
      d = distance(particle, particles[sort_order[ihigh]]);
      while (std::abs(d) <= particle._smoothing_length) {
        do_flux(particle, particles[sort_order[ihigh]], d, solver, gamma, dt);
        // update index
        ihigh = (ihigh < sort_order.size() - 1) ? ihigh + 1 : 0;
        d = distance(particle, particles[sort_order[ihigh]]);
      }
    }

    // move the particles
    for (size_t i = 0; i < particles.size(); ++i) {
      Particle &particle = particles[i];
      particle._position += dt * particle._particle_velocity;
    }
  }

  // output the final snapshot
  dump(particles, std::cout);

  return 0;
}
