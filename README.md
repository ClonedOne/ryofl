# ryofl
(don't) roll your own federated learning


## Desiderata

There are a number of characteristics, that we would like to have in a Federated
Learning framework, that are not really covered in existing frameworks.

Some of the objectives are:

 - *Keep it simple* It should be **really** easy to use this code. It should
   also be equally simple to develop variations starting from this code base.

 - *Actual clients* Simulate each client individually.

 - *Run everywhere* Deploy on multiple small machines.

There are also aspects of the simulation that we would like to make more
realistic, such as:

 - *Rolling window on data* Local data shouldn't (always) be considered stable.
   In a realistic setting local data would become stale over time, and new data
   points would be added periodically.


## Setup

Before continuing please make sure you have setup the correct paths in
`ryofl/common.py`.

## Testing

Run tests with `pytest`.
Tests will be automatically discovered by looking in the `tests/` directory for
files and functions having `test` as a prefix.

## Data

Use `python ryofl/generate_datasets.py` to download the datasets. The location
where to download the datasets can be changed in the header of the file. 

**Note this is the only part of the code which depends on Tensorflow**


