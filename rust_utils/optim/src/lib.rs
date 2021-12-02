#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

// Design principles
//
// 1. data structure
// 2. update the data
//
// Have our optim utils take a struct that points to the object we'll work on
// and implements a trait that allows the optimizer to call `objective_fn`, which
// is exposed to our struct by the implementation of the trait-ish....
//
