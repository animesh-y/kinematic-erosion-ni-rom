#include "udf.h"
#include "dpm.h"
#include <stdio.h>

DEFINE_DPM_BC(custom_rebound, p, t, f, f_normal, dim)
{
    real v_normal = 0.0;
    real v_mag = 0.0;
    real theta = 0.0;
    real en, et;
    int i;

    /* Calculate normal velocity component */
    for (i = 0; i < dim; i++)
    {
        v_normal += p->state.V[i] * f_normal[i];
    }

    v_mag = NV_MAG(p->state.V);

    /* Abort if particle is effectively stopped */
    if (v_mag < 1e-12)
        return PATH_ABORT;

    /* Impact angle relative to wall tangent (in RADIANS) */
    theta = asin(MAX(0.0, MIN(1.0, fabs(v_normal) / v_mag)));

    /* NOTE: If your polynomial coefficients expect DEGREES, uncomment the line below */
    /* real theta_deg = theta * 180.0 / M_PI; and replace 'theta' with 'theta_deg' below */

    /* Calculate Restitution Coefficients */
    en = 0.98913 - 2.0801 * theta + 1.9351 * pow(theta, 2) - 0.51128 * pow(theta, 3);
    et = 1.0104 - 1.4035 * theta + 1.5975 * pow(theta, 2) - 0.44414 * pow(theta, 3);

    /* Bound coefficients to prevent numerical explosion */
    en = MAX(0.0, MIN(1.0, en));
    et = MAX(0.0, MIN(1.0, et));

    /* Write to CSV only from compute nodes */
    // #if !RP_HOST
    //     {
    //         char filename[256];
    //         FILE *fp;

    //         sprintf(filename, "impacts_node_%d.csv", myid);
    //         fp = fopen(filename, "a");
    //         if (fp != NULL)
    //         {
    //             /* Changed wall_name (%s) to wall_id (%d) to prevent node memory crash */
    //             /* x, y, z, impact_angle_rad, impact_velocity, wall_id */
    //             fprintf(fp, "%.12g,%.12g,%.12g,%.12g,%.12g,%d\n",
    //                     p->state.pos[0], p->state.pos[1], p->state.pos[2],
    //                     theta, v_mag, t->id);
    //             fclose(fp);
    //         }
    //     }
    // #endif

    //     /* Calculate and apply new rebound velocity vector */
    //     for (i = 0; i < dim; i++)
    //     {
    //         real v_n_vec = v_normal * f_normal[i];
    //         real v_t_vec = p->state.V[i] - v_n_vec;

    //         p->state.V[i] = -en * v_n_vec + et * v_t_vec;
    //     }

    return PATH_ACTIVE;
}
