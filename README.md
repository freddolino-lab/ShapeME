# Shape-based motif finder

This is the rust development branch. The original version of the code was moved
into the `python3` directory. A new `cargo` created directory `rust` is where
all conversion of the `python3` code into `rust` code will occur.

When adding new code, create a branch from this `rust-dev` branch and submit
pull-requests. The `rust-dev` branch will be considering our working draft.

The first and key aspect of the code that needs converted are the underlying
classes for representing motifs and sequence databases. Everything else is
built on top of this.
