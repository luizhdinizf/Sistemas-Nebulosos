"""
==================================
The Tipping Problem - The Hard Way
==================================

 Note: This method computes everything by hand, step by step. For most people,
 the new API for fuzzy systems will be preferable. The same problem is solved
 with the new API `in this example <./plot_tipping_problem_newapi.html>`_.

The 'tipping problem' is commonly used to illustrate the power of fuzzy logic
principles to generate complex behavior from a compact, intuitive set of
expert rules.

Input variables
---------------

A number of variables play into the decision about how much to tip while
dining. Consider two of them:

* ``quality`` : Quality of the food
* ``service`` : Quality of the service

Output variable
---------------

The output variable is simply the tip amount, in percentage points:

* ``tip`` : Percent of bill to add as tip


For the purposes of discussion, let's say we need 'high', 'medium', and 'low'
membership functions for both input variables and our output variable. These
are defined in scikit-fuzzy as follows

"""
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points

x_qual = np.arange(0, 11, 1)
x_serv = np.arange(0, 11, 1)
x_tip  = np.arange(0, 26, 1)

x = np.arange(-4, 5, 1)
y_real = np.square(x)

y = np.arange(0, 17, 1)


x_proximo_4 = fuzz.trimf(x, [3, 4, 5])
x_proximo_3 = fuzz.trimf(x, [2, 3, 4])
x_proximo_2 = fuzz.trimf(x, [1, 2, 3])
x_proximo_1 = fuzz.trimf(x, [0, 1, 2])
x_proximo_0 = fuzz.trimf(x, [-1, 0, 1])
x_proximo_menos_1 = fuzz.trimf(x, [-2, -1, 0])
x_proximo_menos_2 = fuzz.trimf(x, [-3, -2, -1])
x_proximo_menos_3 = fuzz.trimf(x, [-4, -3, -2])
x_proximo_menos_4 = fuzz.trimf(x, [-5, -4, -3])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))
ax0.plot(x, x_proximo_4, 'r', linewidth=1.5, label='4')
ax0.plot(x, x_proximo_3, 'g', linewidth=1.5, label='3')
ax0.plot(x, x_proximo_2, 'b', linewidth=1.5, label='2')
ax0.plot(x, x_proximo_1, 'r', linewidth=1.5, label='1')
ax0.plot(x, x_proximo_0, 'g', linewidth=1.5, label='0')
ax0.plot(x, x_proximo_menos_1, 'b', linewidth=1.5, label='4')
ax0.plot(x, x_proximo_menos_2, 'r', linewidth=1.5, label='3')
ax0.plot(x, x_proximo_menos_3, 'g', linewidth=1.5, label='2')
ax0.plot(x, x_proximo_menos_4, 'b', linewidth=1.5, label='1')
ax0.set_title('Entradas')
ax0.legend()

y_proximo_16 = fuzz.trimf(y, [9, 16, 16])
y_proximo_9 = fuzz.trimf(y, [4, 9, 16])
y_proximo_4 = fuzz.trimf(y, [1, 4, 9])
y_proximo_1 = fuzz.trimf(y, [0, 1, 4])
y_proximo_0 = fuzz.trimf(y, [0, 0, 1])

ax1.plot(y, y_proximo_16, 'r', linewidth=1.5, label='16')
ax1.plot(y, y_proximo_9, 'g', linewidth=1.5, label='9')
ax1.plot(y, y_proximo_4, 'b', linewidth=1.5, label='4')
ax1.plot(y, y_proximo_1, 'r', linewidth=1.5, label='1')
ax1.plot(y, y_proximo_0, 'g', linewidth=1.5, label='0')




# Now we take our rules and apply them. Rule 1 concerns bad food OR service.
# The OR operator means we take the maximum of these two.
#active_rule0 = np.fmax(x_proximo_0, x_proximo_menos_4)
active_rule0 = x_proximo_0
active_rule1 = np.fmax(x_proximo_1, x_proximo_menos_1)
active_rule2 = np.fmax(x_proximo_2, x_proximo_menos_2)
active_rule3 = np.fmax(x_proximo_3, x_proximo_menos_3)
active_rule4 = np.fmax(x_proximo_4, x_proximo_menos_4)

#
#ax2.plot(x, active_rule1, 'r', linewidth=1.5, label='16')
#ax2.plot(x, active_rule2, 'g', linewidth=1.5, label='16')
#ax2.plot(x, active_rule3, 'b', linewidth=1.5, label='16')
#ax2.plot(x, active_rule4, 'r', linewidth=1.5, label='16')

x_test = -2.5

x_proximo_novo_4 = fuzz.interp_membership(x, active_rule4,x_test)
x_proximo_novo_3 = fuzz.interp_membership(x, active_rule3,x_test)
x_proximo_novo_2 = fuzz.interp_membership(x, active_rule2,x_test)
x_proximo_novo_1 = fuzz.interp_membership(x, active_rule1,x_test)
x_proximo_novo_0 = fuzz.interp_membership(x, active_rule0,x_test)

y_proximo_16_novo =  np.fmin(x_proximo_novo_4, y_proximo_16)
y_proximo_9_novo =  np.fmin(x_proximo_novo_3, y_proximo_9)
y_proximo_4_novo =  np.fmin(x_proximo_novo_2, y_proximo_4)
y_proximo_1_novo =  np.fmin(x_proximo_novo_1, y_proximo_1)
y_proximo_0_novo =  np.fmin(x_proximo_novo_0, y_proximo_0)



saida_fuzzy = np.fmax(y_proximo_16_novo,y_proximo_9_novo)
saida_fuzzy = np.fmax(saida_fuzzy,y_proximo_4_novo)
saida_fuzzy = np.fmax(saida_fuzzy,y_proximo_1_novo)
saida_fuzzy = np.fmax(saida_fuzzy,y_proximo_0_novo)




#
ax2.plot(y, saida_fuzzy, 'g', linewidth=1.5, label='16')


# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
#tip_activation_lo = np.fmin(active_rule1, tip_lo)  # removed entirely to 0
#
## For rule 2 we connect acceptable service to medium tipping
#tip_activation_md = np.fmin(serv_level_md, tip_md)
#
## For rule 3 we connect high service OR high food with high tipping
#active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
#tip_activation_hi = np.fmin(active_rule3, tip_hi)
#tip0 = np.zeros_like(x_tip)

# Visualize this
#fig, ax0 = plt.subplots(figsize=(8, 3))

#ax0.fill_between(x_tip, tip0, tip_activation_lo, facecolor='b', alpha=0.7)
#ax0.plot(x_tip, tip_lo, 'b', linewidth=0.5, linestyle='--', )
#ax0.fill_between(x_tip, tip0, tip_activation_md, facecolor='g', alpha=0.7)
#ax0.plot(x_tip, tip_md, 'g', linewidth=0.5, linestyle='--')
#ax0.fill_between(x_tip, tip0, tip_activation_hi, facecolor='r', alpha=0.7)
#ax0.plot(x_tip, tip_hi, 'r', linewidth=0.5, linestyle='--')
#ax0.set_title('Output membership activity')
#
## Turn off top/right axes
#for ax in (ax0,):
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.get_xaxis().tick_bottom()
#    ax.get_yaxis().tick_left()
#
#plt.tight_layout()

"""
.. image:: PLOT2RST.current_figure

Rule aggregation
----------------

With the *activity* of each output membership function known, all output
membership functions must be combined. This is typically done using a
maximum operator. This step is also known as *aggregation*.

Defuzzification
---------------
Finally, to get a real world answer, we return to *crisp* logic from the
world of fuzzy membership functions. For the purposes of this example
the centroid method will be used.

The result is a tip of **20.2%**.
---------------------------------
"""

# Aggregate all three output membership functions together
#aggregated = np.fmax(tip_activation_lo,
#                     np.fmax(tip_activation_md, tip_activation_hi))
#
## Calculate defuzzified result
#tip = fuzz.defuzz(x_tip, aggregated, 'centroid')
#tip_activation = fuzz.interp_membership(x_tip, aggregated, tip)  # for plot

# Visualize this
#fig, ax0 = plt.subplots(figsize=(8, 3))
#
#ax0.plot(x_tip, tip_lo, 'b', linewidth=0.5, linestyle='--', )
#ax0.plot(x_tip, tip_md, 'g', linewidth=0.5, linestyle='--')
#ax0.plot(x_tip, tip_hi, 'r', linewidth=0.5, linestyle='--')
#ax0.fill_between(x_tip, tip0, aggregated, facecolor='Orange', alpha=0.7)
#ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
#ax0.set_title('Aggregated membership and result (line)')
#
## Turn off top/right axes
#for ax in (ax0,):
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.get_xaxis().tick_bottom()
#    ax.get_yaxis().tick_left()
#
#plt.tight_layout()

"""
.. image:: PLOT2RST.current_figure

Final thoughts
--------------

The power of fuzzy systems is allowing complicated, intuitive behavior based
on a sparse system of rules with minimal overhead. Note our membership
function universes were coarse, only defined at the integers, but
``fuzz.interp_membership`` allowed the effective resolution to increase on
demand. This system can respond to arbitrarily small changes in inputs,
and the processing burden is minimal.

"""