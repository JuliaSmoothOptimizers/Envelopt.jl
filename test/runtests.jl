using TestItemRunner

@run_package_tests verbose=true

# filter tests with
# @run_package_tests verbose=true filter=ti -> (:trunk in ti.tags || :tron in ti.tags)
