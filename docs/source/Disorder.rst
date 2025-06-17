Disorder
========

When reading disordered structures, the package will try to determine
the disorder assemblies and groups, which will be stored in the ``.disorder``
attribute of the :class:`PeriodicSet <amd.periodicset.PeriodicSet>`. When
calculating an invariant of a disordered crystal, we create one invariant
for each disorder configuration, which are then concatenated into one,
which maintains invariance and continuity (even if there are a different
number of groups between two crystals but the geometry is similar).

Therefore, calculating PDD can be significantly slower for disordered
structures than ordered ones, and there is a cap on the number of disorder
configurations that the package will allow before defaulting to the
maximum occupancy configuration, set by
:data:`amd.globals.MAX_DISORDER_CONFIGS <amd.globals.MAX_DISORDER_CONFIGS>`
with the default value 100.

Since multiple PDDs are calculated for each config, the final PDD may
have multiple rows corresponding to a single point, as opposed to an ordered
structure where at most one row corresponds to one point.

Handling disorder is a relatively new aspect of the package; is not yet
well-tested and may give unexpected results in some cases. It is preferable
to specifiy the disorder assemblies and groups through the CIF tags
'_atom_site_disorder_assembly' and '_atom_site_disorder_group', if they are
not found the package will try to assemble groups for you but the correct
groups may not always be found.
