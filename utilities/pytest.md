

# Introduction

Suppose we have the following `test_one.py` file:


```python

def test_passing():

    assert (1,2,3) == (1,2,3)


def test_failing():

    assert (1,2,3) == (3,2,1)

```

Pytest command:

```python

pytest test_one.py

```

If you need more information, you can use `-v` or `--verbose`

```python

pytest -v test_one.py

```

Other flags:

* `--tb=no`: Turn off tracebacks
* `-v`: Verbose
* `-vv`: Extra verbose

To run pytest, you have the option to specify files and directories. If you don’t specify any files or directories, pytest will look for tests in the current working directory and subdirectories. It looks for .py files starting with `test_` or ending with `_test`.

We can also specify a test function within a test file to run by adding ::test_name to the file name: ​  

```python

​pytest​​ ​​-v​​ ​​test_one.py::test_passing

```


Given no arguments, pytest looks at your current directory and all subdirectories for test files and runs the test code it finds. If you give pytest a filename, a directory name, or a list of those, it looks there instead of the current directory. Each directory listed on the command line is examined for test code, as well as any subdirectories. Here’s a brief overview of the naming conventions to keep your test code discoverable by pytest:

-Test files should be named `test_<something>.py` or `<something>_test.py`.
-Test methods and functions should be named `test_<something>`.
-Test classes should be named `Test<Something>.`


## Structuring Test Functions

Keep assertions at the end of test functions. This is such a common recommendation that it has at least two names: Arrange-Act-Assert and Given-When-Then.

A common anti-pattern is to have more a “Arrange-Assert-Act-Assert-Act-Assert…” pattern where lots of actions, followed by state or behavior checks, validate a workflow. This seems reasonable until the test fails. Any of the actions could have caused the failure, so the test is not focusing on testing one behavior. Or it might have been the setup in “Arrange” that caused the failure. This interleaved assert pattern creates tests that are hard to debug and maintain because later developers have no idea what the original intent of the test was. Sticking to Given-When-Then or Arrange-Act-Assert keeps the test focused and makes the test more maintainable.

Let’s apply this structure to one of our first tests as an example:

```python
def​ ​test_to_dict​(): ​  ​
	# GIVEN a Card object with known contents
	c1 = Card(​"something"​, ​"brian"​, ​"todo"​, 123) ​  ​  ​
	
	# WHEN we call to_dict() on the object​ ​ 
	c2 = c1.to_dict() ​  ​  ​
	
	# THEN the result will be a dictionary with known content​ ​  
	c2_expected = { ​  ​"summary"​: ​"something"​,
	 ​  ​"owner"​: ​"brian"​,
	  ​  ​"state"​: ​"todo"​,
	   ​  ​"id"​: 123, ​  
	   } ​  ​
	assert​ c2 == c2_expected
```


- Given/Arrange—A starting state. This is where you set up data or the environment to get ready for the action. 
- When/Act—Some action is performed. This is the focus of the test—the behavior we are trying to make sure is working right. 
- Then/Assert—Some expected result or end state should happen. At the end of the test, we make sure the action resulted in the expected behavior.


## Test Outcomes

Here are the possible outcomes of a test:

- PASSED (.)—The test ran successfully.

- FAILED (F)—The test did not run successfully.

- SKIPPED (s)—The test was skipped. You can tell pytest to skip a test by using either the @pytest.mark.skip() or @pytest.mark.skipif() decorators

- XFAIL (x)—The test was not supposed to pass, and it ran and failed. You can tell pytest that a test is expected to fail by using the @pytest.mark.xfail() decorator, which is discussed in ​Expecting Tests to Fail with pytest.mark.xfail​.

- XPASS (X)—The test was marked with xfail, but it ran and passed.

- ERROR (E)—An exception happened either during the execution of a fixture or hook function, and not during the execution of a test function.


## Commands

### pytest.fail()

A test will fail if there is any uncaught exception. This can happen if
- an assert statement fails, which will raise an AssertionError exception, 
- the test code calls pytest.fail(), which will raise an exception, 
- or any other exception is raised. 

While any exception can fail a test, I prefer to use assert. In rare cases where assert is not suitable, use pytest.fail(). Here’s an example of using pytest’s fail() function to explicitly fail a test:

```python
import​ ​pytest​ ​  
​from​ ​cards​ ​import​ Card ​  ​  ​  ​

def​ ​test_with_fail​(): ​ 
	c1 = Card(​"sit there"​, ​"brian"​) ​  
	c2 = Card(​"do something"​, ​"okken"​) ​  ​
	if​ c1 != c2: ​  
		pytest.fail(​"they don't match"​)
```

When calling pytest.fail() or raising an exception directly, we don’t get the wonderful assert rewriting provided by pytest. However, there are reasonable times to use pytest.fail(), such as in an assertion helper.

### pytest.raises()

What if a bit of code you are testing is supposed to raise an exception? How do you test for that? You use pytest.raises() to test for expected exceptions.

```python
from​ ​cards​ ​import​ Card ​  
​import​ ​pytest​ ​  ​  ​  

​def​ ​test_no_path_raises​(): ​  
	​with​ pytest.raises(TypeError): ​  
		cards.CardsDB()
```

The `with pytest.raises(TypeError):` statement says that whatever is in the next block of code should raise a TypeError exception. If no exception is raised, the test fails. If the test raises a different exception, it fails. 

We just checked for the type of exception in test_no_path_raises(). We can also check to make sure the message is correct, or any other aspect of the exception, like additional parameters:

```python
from​ ​cards​ ​import​ Card ​  
​import​ ​pytest​ ​  ​  ​  

​def​ ​test_raises_with_info​(): ​  
	match_regex = ​"missing 1 .* positional argument"​ ​  
	​with​ pytest.raises(TypeError, match=match_regex): ​  
		cards.CardsDB() ​  ​  ​  ​
		
def​ ​test_raises_with_info_alt​(): ​  
	​with​ pytest.raises(TypeError) ​as​ exc_info: ​  
		cards.CardsDB() ​  
	expected = ​"missing 1 required positional argument"​ ​  ​
	assert​ expected ​in​ str(exc_info.value)
```

The match parameter takes a regular expression and matches it with the exception message. You can also use as exc_info or any other variable name to interrogate extra parameters to the exception if it’s a custom exception. The exc_info object will be of type ExceptionInfo.


### Assertion Helper Function

An assertion helper is a function that is used to wrap up a complicated assertion check. As an example, the Cards data class is set up such that two cards with different IDs will still report equality. If we wanted to have a stricter check, we could write a helper function called assert_identical like this:

```python
from​ ​cards​ ​import​ Card ​  
​import​ ​pytest​ ​  ​  ​  

​def​ ​assert_identical​(c1: Card, c2: Card): ​  
	__tracebackhide__ = True ​  ​
	assert​ c1 == c2 ​  # Evals to True even if ids are different​
	if​ c1.id != c2.id: ​  
		pytest.fail(f​"id's don't match. {c1.id} != {c2.id}"​) ​  ​  ​ 

​def​ ​test_identical​(): ​  
	c1 = Card(​"foo"​, id=123) ​  
	c2 = Card(​"foo"​, id=123) ​  
	assert_identical(c1, c2)
```

The assert_identical function sets`__tracebackhide__ = True`. This is optional. The effect will be that failing tests will not include this function in the traceback. The normal assert c1 == c2 is then used to check everything except the ID for equality. Finally, the IDs are checked, and if they are not equal, pytest.fail() is used to fail the test with a hopefully helpful message.

I could have also used assert c1.id == c2.id, "id’s don’t match." to much the same effect.


# Fixtures

Fixtures are functions that are run by pytest before (and sometimes after) the actual test functions. The code in the fixture can do whatever you want it to. You can use fixtures to get a data set for the tests to work on. You can use fixtures to get a system into a known state before running a test. Fixtures are also used to get data ready for multiple tests.

Here’s a simple fixture that returns a number:

```python
import​ ​pytest​ ​  ​  ​  

@pytest.fixture() ​  ​
def​ ​some_data​(): ​  ​
	"""Return answer to ultimate question."""​ ​  
	​return​ 42 ​  

​def​ ​test_some_data​(some_data): ​  
	​"""Use fixture return value in a test."""​ ​  
	​assert​ some_data == 42
```

The @pytest.fixture() decorator is used to tell pytest that a function is a fixture. When you include the fixture name in the parameter list of a test function, pytest knows to run it before running the test. Fixtures can do work, and can also return data to the test function.

pytest treats exceptions differently during fixtures compared to during a test function. An exception (or assert failure or call to pytest.fail()) that happens during the test code proper results in a “Fail” result. However, during a fixture, the test function is reported as “Error.”

Another example:

```python
import​ ​pytest​ ​  ​  ​  

@pytest.fixture() ​  ​
def​ ​cards_db​(): ​  
	​with​ TemporaryDirectory() ​as​ db_dir: ​  
		db_path = Path(db_dir) ​  
		db = cards.CardsDB(db_path) ​  
		​yield​ db ​  
		db.close() ​  ​  ​  
		
​def​ ​test_empty​(cards_db): ​  
	​assert​ cards_db.count() == 0

​​def​ ​test_two​(cards_db): ​  
	cards_db.add_card(cards.Card(​"first"​)) ​  
	cards_db.add_card(cards.Card(​"second"​)) ​  ​
	assert​ cards_db.count() == 2
```

Fixture functions run before the tests that use them. If there is a yield in the function, it stops there, passes control to the tests, and picks up on the next line after the tests are done. The code above the yield is “setup” and the code after yield is “teardown.” The code after the yield, the teardown, is guaranteed to run regardless of what happens during the tests.

The above example also shows you can use fixtures in multiple tests.

pytest provides the command-line flag, --setup-show, which shows us the order of operations of tests and fixtures, including the setup and teardown phases of the fixtures:

```
$ ​​pytest​​ ​​--setup-show​​ ​​test_count.py​ ​  
======================== test session starts ========================= ​  collected 2 items ​  ​  
test_count.py ​  
	SETUP    F cards_db ​  
	ch3/test_count.py::test_empty (fixtures used: cards_db). ​  
	TEARDOWN F cards_db ​  
	SETUP    F cards_db ​  
	ch3/test_count.py::test_two (fixtures used: cards_db). ​  
	TEARDOWN F cards_db ​  ​  
========================= 2 passed in 0.02s ==========================
```

The `F` in front of the fixture name indicates that the fixture is using function scope, meaning the fixture is called before each test function that uses it, and torn down after each function that uses it.

### Fixture Scope

Each fixture has a specific scope, which defines the order of when the setup and teardown run relative to running of all the test function using the fixture. The scope dictates how often the setup and teardown get run when it’s used by multiple test functions. 

The default scope for fixtures is function scope. That means the setup portion of the fixture will run before each test that needs it runs. Likewise, the teardown portion runs after the test is done, for each test. 

The scopes are:

1. **Function scope** (`scope="function"`): This is the default scope. The fixture is invoked once per test function. It is ideal for setting up conditions that should be fresh for each test, ensuring that no state persists across tests that might lead to unexpected results.
    
2. **Class scope** (`scope="class"`): The fixture is invoked once per test class. All methods within that class will use the same instance of the fixture. This is useful when you want to set up a state that is shared across multiple tests in the same class but not across different test classes.
    
3. **Module scope** (`scope="module"`): The fixture is invoked once per module (a Python file). This allows all the tests in a module to share the setup, which can be efficient when setting up the state is time-consuming.
    
4. **Package scope** (`scope="package"`): The fixture is created once for the entire package. Every test in the package can use this fixture. This is less commonly used but can be useful for heavy-weight shared resources across multiple modules in the same package.
    
5. **Session scope** (`scope="session"`): The fixture is invoked once per test session. All tests in all classes and modules during a test run can use this fixture. This is suitable for very expensive operations, like setting up a database or starting up a Docker container, which should be done only once during the entire testing process.

NOTE: Fixtures can only depend on other fixtures of their same scope or wider. So a function-scope fixture can depend on other function-scope fixtures. A function-scope fixture can also depend on class-, module-, and session-scope fixtures, but you can’t go in the reverse order.


Example:

```python
import​ ​pytest​ ​  ​  ​  

@pytest.fixture(scope="module") ​  ​
def​ ​cards_db​(): ​  
	​with​ TemporaryDirectory() ​as​ db_dir: ​  
		db_path = Path(db_dir) ​  
		db = cards.CardsDB(db_path) ​  
		​yield​ db ​  
		db.close() ​  ​  ​  
```


### Dynamic Fixture Scope

To set the scope dynamically, you cannot directly use the `scope` parameter because it expects a static string. Instead, you will need to use a factory pattern to create fixtures with different scopes based on a condition:

```python
import pytest

def determine_scope():
    # Define your logic to determine the scope
    if some_condition:
        return 'session'
    else:
        return 'function'

def resource_setup(scope):
    @pytest.fixture(scope=scope)
    def actual_fixture():
        # Setup code here
        yield
        # Teardown code here
    return actual_fixture

# Creating fixture dynamically
some_resource = resource_setup(determine_scope())

def test_example(some_resource):     
	# your test code using the fixture     
	pass
```


### Using Multiple Fixtures

Consider a modification of an earlier example:

```python
import​ ​pytest​ ​  ​  ​  

@pytest.fixture() ​  ​
def​ ​cards_db​(): ​  
	​with​ TemporaryDirectory() ​as​ db_dir: ​  
		db_path = Path(db_dir) ​  
		db = cards.CardsDB(db_path) ​  
		​yield​ db ​  
		db.close() ​  ​  ​  
		
​def​ ​test_empty​(cards_db): ​  
	​assert​ cards_db.count() == 0

​​def​ ​test_two​(cards_db): ​  
	cards_db.add_card(cards.Card(​"first"​)) ​  
	cards_db.add_card(cards.Card(​"second"​)) ​  ​
	assert​ cards_db.count() == 2

def​ ​test_three​(cards_db): ​  
	cards_db.add_card(cards.Card(​"first"​)) ​  
	cards_db.add_card(cards.Card(​"second"​)) ​  
	cards_db.add_card(cards.Card(​"third"​)) ​  
	​assert​ cards_db.count() == 3
```


Tests shouldn’t rely on the run order. And clearly, this does. test_three passes just fine if we run it by itself, but fails if it is run after test_two. If we still want to try to stick with one open database, but start all the tests with zero elements in the database, we can do that by adding another fixture:

```python
import​ ​pytest​ ​  ​  ​  

@pytest.fixture(scope=​"session"​) ​  
​def​ ​db​(): ​ 
	​"""CardsDB object connected to a temporary database"""​ ​  
	​with​ TemporaryDirectory() ​as​ db_dir: ​  
		db_path = Path(db_dir) ​  
		db_ = cards.CardsDB(db_path)
		yield db_
		db_.close()

@pytest.fixture(scope=​"function"​) ​  ​
def​ ​cards_db​(db): ​  
	​"""CardsDB object that's empty"""​ ​  
	db.delete_all() ​  
	​return​ db 
		
​def​ ​test_empty​(cards_db): ​  
	​assert​ cards_db.count() == 0

​​def​ ​test_two​(cards_db): ​  
	cards_db.add_card(cards.Card(​"first"​)) ​  
	cards_db.add_card(cards.Card(​"second"​)) ​  ​
	assert​ cards_db.count() == 2

def​ ​test_three​(cards_db): ​  
	cards_db.add_card(cards.Card(​"first"​)) ​  
	cards_db.add_card(cards.Card(​"second"​)) ​  
	cards_db.add_card(cards.Card(​"third"​)) ​  
	​assert​ cards_db.count() == 3
```

We can see that the setup for db happens first, and has session scope. The setup for cards_db happens next, and before each test function call, and has function scope.


Another way we can use multiple fixtures is just to use more than one in either a function or a fixture.

```python
import​ ​pytest​ ​  ​  ​  

@pytest.fixture(scope=​"session"​) ​  
​def​ ​db​(): ​ 
	​"""CardsDB object connected to a temporary database"""​ ​  
	​with​ TemporaryDirectory() ​as​ db_dir: ​  
		db_path = Path(db_dir) ​  
		db_ = cards.CardsDB(db_path)
		yield db_
		db_.close()

@pytest.fixture(scope=​"function"​) ​  ​
def​ ​cards_db​(db): ​  
	​"""CardsDB object that's empty"""​ ​  
	db.delete_all() ​  
	​return​ db 
	
@pytest.fixture(scope=​"session"​) ​  ​
def​ ​some_cards​(): ​  ​
	"""List of different Card objects"""​ ​  
	​return​ [ ​  cards.Card(​"write book"​, ​"Brian"​, ​"done"​),
	 ​  cards.Card(​"edit book"​, ​"Katie"​, ​"done"​),
	  ​  cards.Card(​"write 2nd edition"​, ​"Brian"​, ​"todo"​),
	   ​  cards.Card(​"edit 2nd edition"​, ​"Katie"​, ​"todo"​),​  ]

def​ ​test_add_some​(cards_db, some_cards): ​  
	expected_count = len(some_cards) ​  ​
	for​ c ​in​ some_cards: ​  
		cards_db.add_card(c) ​  ​
	assert​ cards_db.count() == expected_count
```


Fixtures can also use multiple other fixtures:

```python
@pytest.fixture(scope=​"function"​) ​  
​def​ ​non_empty_db​(cards_db, some_cards): ​  ​
	"""CardsDB object that's been populated with 'some_cards'"""​ ​  ​
	for​ c ​in​ some_cards: ​  
		cards_db.add_card(c) ​  
	​return​ cards_db
```


### Sharing Fixtures through conftest.py

You can put fixtures into individual test files, but to share fixtures among multiple test files, you need to use a conftest.py file either in the same directory as the test file that’s using it or in some parent directory. The conftest.py file is also optional. It is considered by pytest as a “local plugin” and can contain hook functions and fixtures.

Although conftest.py is a Python module, it should not be imported by test files. The conftest.py file gets read by pytest automatically, so you don’t have import conftest anywhere.

### Finding Where Fixtures are Defined

If we can’t remember where a particular fixture is located and we want to see the source code, just use --fixtures:

```
pytest --fixtures -v
```

You can also use --fixtures-per-test to see what fixtures are used by each test and where the fixtures are defined:

```
pytest​​ ​​--fixtures-per-test​​ ​​test_count.py::test_empty
```

In this example we’ve specified an individual test, test_count.py::test_empty. However, the flag works for files or directories as well.


### Built-in Fixtures

Reusing common fixtures is such a good idea that the pytest developers included some commonly used fixtures with pytest. The builtin fixtures that come prepackaged with pytest can help you do some pretty useful things in your tests easily and consistently.

The tmp_path and tmp_path_factory fixtures are used to create temporary directories. The tmp_path function-scope fixture returns a pathlib.Path instance that points to a temporary directory that sticks around during your test and a bit longer. The tmp_path_factory session-scope fixture returns a TempPathFactory object. This object has a mktemp() function that returns Path objects. You can use mktemp() to create multiple temporary directories. You use them like this:

```python
def​ ​test_tmp_path​(tmp_path): ​  
	file = tmp_path / ​"file.txt"​ ​  
	file.write_text(​"Hello"​) ​  ​
	assert​ file.read_text() == ​"Hello"​ ​  ​  ​  

​def​ ​test_tmp_path_factory​(tmp_path_factory): 
	path = tmp_path_factory.mktemp(​"sub"​) ​  
	file = path / ​"file.txt"​ ​  
	file.write_text(​"Hello"​) ​  ​
	assert​ file.read_text() == ​"Hello"​
```

Note: `tmp_path_factory` is session scope, while `tmp_path` is function scope.

Other built-in fixtures include:
- capsys—for capturing output 
- monkeypatch—for changing the environment or application code, like a lightweight form of mocking
- capfd, capfdbinary, capsysbinary—Variants of capsys that work with file descriptors and/or binary output 
- caplog—Similar to capsys and the like; used for messages created with Python’s logging system 
- cache—Used to store and retrieve values across pytest runs. The most useful part of this fixture is that it allows for --last-failed, --failed-first, and similar flags. 
- doctest_namespace—Useful if you like to use pytest to run doctest-style tests 
- pytestconfig—Used to get access to configuration values, pluginmanager, and plugin hooks 
- record_property, record_testsuite_property—Used to add extra properties to the test or test suite. Especially useful for adding data to an XML report to be used by continuous integration tools 
- recwarn—Used to test warning messages 
- request—Used to provide information on the executing test function. Most commonly used during fixture parametrization 
- pytester, testdir—Used to provide a temporary test directory to aid in running and testing pytest plugins. pytester is the pathlib based replacement for the py.path based testdir.


# Parameterization

Parametrized testing refers to adding parameters to our test functions and passing in multiple sets of arguments to the test to create new test cases. We’ll look at three ways to implement parametrized testing in pytest in the order in which they should be selected: 

- Parametrizing functions 
- Parametrizing fixtures 
- Using a hook function called pytest_generate_tests

### Parameterizing Functions

To parametrize a test function, add parameters to the test definition and use the @pytest.mark.parametrize() decorator to define the sets of arguments to pass to the test, like this: 

```python
import​ ​pytest​ ​  
​from​ ​cards​ ​import​ Card ​  ​  ​ 

@pytest.mark.parametrize( ​  ​
	"start_summary, start_state"​,
	[ ​ (​"write a book"​, ​"done"​),
	 ​  (​"second edition"​, ​"in prog"​),
	  ​ (​"create a course"​, ​"todo"​), ​  
	  ], ​  
) ​  
​def​ ​test_finish​(cards_db, start_summary, start_state): ​  
	initial_card = Card(summary=start_summary, state=start_state) ​  
	index = cards_db.add_card(initial_card) ​  ​  
	cards_db.finish(index) ​  ​  
	card = cards_db.get_card(index)
	assert card.state == "done"
```


The test_finish() function now has its original cards_db fixture as a parameter, but also two new parameters: start_summary and start_state. These match directly to the first argument to @pytest.mark.parametrize(). The first argument to @pytest.mark.parametrize() is a list of names of the parameters. They are strings and can be an actual list of strings, as in ["start_summary", "start_state"], or they can be a comma-separated string, as in "start_summary, start_state". The second argument to @pytest.mark.parametrize() is our list of test cases. Each element in the list is a test case represented by a tuple or list that has one element for each argument that gets sent to the test function. pytest will run this test once for each (start_summary, start_state) pair and report each as a separate test,

### Parameterizing Fixtures

When we used function parametrization, pytest called our test function once each for every set of argument values we provided. With fixture parametrization, we shift those parameters to a fixture. pytest will then call the fixture once each for every set of values we provide. Then downstream, every test function that depends on the fixture will be called, once each for every fixture value. Also, the syntax is different:

```python
@pytest.fixture(params=[​"done"​, ​"in prog"​, ​"todo"​]) ​  
​def​ ​start_state​(request): ​  ​
	return​ request.param ​  ​  ​  

​def​ ​test_finish​(cards_db, start_state):
	...
```


### Parametrizing with pytest_generate_tests 

The third way to parametrize is by using a hook function called `pytest_generate_tests`.


## Markers

In pytest, markers are a way to tell pytest there’s something special about a particular test. You can think of them like tags or labels. If some tests are slow, you can mark them with @pytest.mark.slow and have pytest skip those tests when you’re in a hurry. You can pick a handful of tests out of a test suite and mark them with @pytest.mark.smoke and run those as the first stage of a testing pipeline in a continuous integration system. Really, for any reason you might have for separating out some tests, you can use markers.

pytest’s builtin markers are used to modify the behavior of how tests run. Here’s the full list of the builtin markers: 
- @pytest.mark.filterwarnings(warning): This marker adds a warning filter to the given test. 
- @pytest.mark.skip(reason=None): This marker skips the test with an optional reason. 
- @pytest.mark.skipif(condition, ..., *, reason): This marker skips the test if any of the conditions are True. 
- @pytest.mark.xfail(condition, ..., *, reason, run=True, raises=None, strict=xfail_strict): This marker tells pytest that we expect the test to fail. 
- @pytest.mark.parametrize(argnames, argvalues, indirect, ids, scope): This marker calls a test function multiple times, passing in different arguments in turn. 
- @pytest.mark.usefixtures(fixturename1, fixturename2, ...): This marker marks tests as needing all the specified fixtures. These are the most commonly used of these builtins: 
- @pytest.mark.parametrize() 
- @pytest.mark.skip() 
- @pytest.mark.skipif() 
- @pytest.mark.xfail()


# pytest.ini File

The `pytest.ini` file is a configuration file used by `pytest`, a popular Python testing framework. It allows you to specify various settings that control the behavior of `pytest` across your entire project.

Here are some common settings that can be specified in `pytest.ini`:

1. **Test Path**: Define where `pytest` should look for test files.
2. **Python Files**: Specify patterns that `pytest` will recognize as test files.
3. **Test Functions and Classes**: Define how test functions and classes are identified.
4. **Plugins**: Manage plugins, such as enabling and disabling them.
5. **Markers**: Declare custom markers, which can be used to tag tests for selective running.
6. **Options**: Set default command line options to be applied each time you run `pytest`.

Here’s a basic example:

```ini
[pytest]
minversion = 6.0
addopts = -ra -q
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: mark tests as slow running
    integration: mark tests as integration tests
```

- **minversion**: Specifies the minimum version of `pytest` required to run the tests.
- **addopts**: Adds default options to the `pytest` command line. For example, `-ra` provides a summary of test outcomes, and `-q` runs `pytest` in quiet mode.
- **testpaths**: Lists directories that `pytest` should search for tests.
- **python_files**: Patterns to match test file names.
- **python_classes**: Patterns to match test class names.
- ***python_functions**: Patterns to match test function names.
- **markers**: Custom markers can be declared under this option, which can then be used with the `-m` flag to selectively run tests.

When you run `pytest`, it will automatically look for a `pytest.ini`, `tox.ini`, or `pyproject.toml` file in your project directory to load these configurations. If `pytest.ini` exists, it takes precedence over the others for `pytest` configurations.

The `pytest.ini` file should be placed at the root of your project directory. This is the directory from which you typically invoke the `pytest` command. Placing it at the root ensures that `pytest` can easily find and read the configuration settings when it starts.

# Testing a Simple Python Script

Let’s start with the canonical coding example, *Hello World!:

```python
ch12/script/hello.py

print("Hello, World!")
````

The run output shouldn’t be surprising:

```bash
$ cd /path/to/code/ch12/script
$ python hello.py
Hello, World!
```

Like any other bit of software, scripts are tested by running them and checking the output and/or side effects.

For the `hello.py` script, our challenge is to (1) figure out how to run it from a test, and (2) how to capture the output. The `subprocess` module, which is part of the Python standard library, has a `run()` method that will solve both problems just fine:

```python
ch12/script/test_hello.py

from subprocess import run

def test_hello():
    result = run(["python", "hello.py"], capture_output=True, text=True)
    output = result.stdout
    assert output == "Hello, World!\n"
```

The test launches a subprocess, captures the output, and compares it to "Hello, World!\n", including the newline print() automatically adds to the output. Let’s try it out:

```
pytest -v test_hello.py
```


Suppose we have a bunch of scripts and a bunch of tests for those scripts, and our directory is getting a bit cluttered. So we decide to move the scripts into a `src` directory and the tests into a `tests` directory, like this:

```
script_src 
├── src 
│ └── hello.py 
├── tests
│ └── test_hello.py 
└── pytest.ini
```


Without any other changes, pytest will blow up:

```bash
$ cd /path/to/code/ch12/script_src
$ pytest --tb=short -c pytest_bad.ini
````

Our tests—and pytest—don’t know to look in src for hello. All import statements, either in our source code or in our test code, use the standard Python import process; therefore, they look in directories that are found in the Python module search path. Python keeps this search path list in the sys.path variable, then pytest modifies this list a bit to add the directories of the tests it’s going to run. What we need to do is add the directories for the source code we want to import into sys.path. pytest has an option to help us with that, pythonpath. The option was introduced for pytest 7.

First we need to modify our pytest.ini to set pythonpath to src:

```
ch12/script_src/pytest.ini ​  ​
[pytest]​ ​  
addopts = ​-ra​ ​  
testpaths = ​tests​
pythonpath = ​src​
```

Now pytest runs just fine:

```
pytest tests/test_hello.py
```

