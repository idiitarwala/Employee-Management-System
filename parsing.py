"""
File containing relevant REPL/AST classes and methods.
"""
from typing import List, Any, Optional
import graph


def remove_whitespace(s: str) -> str:
    """
    Remove whitespace from a string.

    >>> s = ' this is a string with whitespace '
    >>> remove_whitespace(s) == 'thisisastringwithwhitespace'
    True
    """
    return ''.join(s.split(' '))


def tokenize(s: str) -> List[str]:
    """
    Tokenize a string by splitting on whitespace except when contained in ()'s.
    If there contains a balanced bracket, split it as well.

    >>> s = ' i want to tokenize (this string) '
    >>> tokenize(s) == ['i', 'want', 'to', 'tokenize', '(this string)']
    True
    """
    tokens = []
    medium = []
    j = 0
    for i in range(0, len(s)):
        if s[i] == '(':
            medium.append(s[j:i])
            j = i
        if s[i] == ')':
            medium.append(s[j:i + 1])
            j = i + 1
    if j < len(s):
        medium.append(s[j:])
    for x in medium:
        x = x.strip()
        if len(x) == 0:
            continue
        if x[0] == '(':
            tokens.append(x)
        else:
            tokens.extend(x.split(' '))
    return tokens


class Expr:
    """
    Expression interface for AST.
    """

    def evaluate(self) -> Any:
        """
        Evaluate the expression.
        """
        raise NotImplementedError


class UnOp(Expr):
    """
    Class representing an unary operator.

    Instance Attributes:
        - company: Underlying graph structure (in this case a tree) modelling the company
        - op: String denoting unary operator
        - right: Expression that operator is acting upon
    """
    company: graph.Tree
    op: str
    right: Expr

    def __init__(self, company: graph.Tree, op: str, right: Expr) -> None:
        self.company = company
        self.op = op
        self.right = right

    def evaluate(self) -> Any:
        """
        Evaluate unary operator.
        """
        try:
            if self.op == 'BOSS':
                self.company.add_root(self.right.evaluate())
            if self.op == 'DEMOTE':
                self.company.demote_vertex(self.right.evaluate().uid)
            if self.op == 'FIRE':
                self.company.delete_vertex(self.right.evaluate().uid)
        except AttributeError:
            pass


class BinOp(Expr):
    """
    Class representing binary operator.

    Instance Attributes:
        - company: Underlying graph structure (in this case a tree) modelling the company
        - left: Expression that operator is acting upon (first input)
        - op: String denoting unary operator
        - right: Expression that operator is acting upon (second input)
    """
    company: graph.Tree
    left: Expr
    op: str
    right: Expr

    def __init__(self, company: graph.Tree, left: Expr, op: str, right: Expr) -> None:
        self.company = company
        self.left = left
        self.op = op
        self.right = right

    def evaluate(self) -> Any:
        """
        Evaluate binary operator.
        """
        if self.op == 'ADD':
            potentially_new = self.left.evaluate()
            if not self.company.has(potentially_new.uid):
                self.company.add_vertex(potentially_new)
            self.company.add_edge(potentially_new.uid, self.right.evaluate().uid)
        if self.op == 'RESOLVE':
            lca = self.company.lca(self.left.evaluate().uid, self.right.evaluate().uid)
            return str(Employee(self.company, str(lca)).evaluate().uid)
        if self.op == 'UPDATE':
            self.company.update_vertex_performance(self.left.evaluate().uid, self.right.evaluate())


class Employee(Expr):
    """
    Class representing an Employee in the company.

    Instance Attributes:
        - company: Underlying graph structure modelling the company
        - token: Input token used to refer to employee
    """
    company: graph.Tree
    token: str

    def __init__(self, company: graph.Tree, token: str) -> None:
        self.company = company
        self.token = token

    def evaluate(self) -> graph.Vertex:
        """
        Evaluate employee.
        Should always evaluate to a Vertex object.
        """
        if self.token.isdigit():
            return self.company.get(int(self.token))
        else:
            # In the form (name, performance), where performance optional
            self.token = remove_whitespace(self.token)
            self.token = self.token.lstrip('(').rstrip(')')
            tokens = self.token.split(',')
            if len(tokens) == 1:
                return graph.Vertex(tokens[0], self.company.next_uid)
            if len(tokens) == 2:
                return graph.Vertex(tokens[0], self.company.next_uid, int(tokens[1]))


class Num(Expr):
    """
    Class representing a number.

    Instance Attributes:
        - token: Input token to denote this number
    """
    token: str

    def __init__(self, token: str) -> None:
        self.token = token

    def evaluate(self) -> Optional[int]:
        """
        Evaluate number.
        """
        if not self.token.isdigit():
            print(self.token + ' is not a valid integer.')
            return
        return int(self.token)


class REPL:
    """
    REPL which allows the user to interact with the program.
    """
    # Private Instance Attributes:
    #     - _company: Underlying graph structure to represent _company
    #     - _buffer: Input buffer that is yet to be tokenized
    #     - _module: AST module
    _company: graph.Tree
    _buffer: str
    _module: List[Expr]

    def __init__(self, company: graph.Tree) -> None:
        self._company = company
        self._buffer = ''
        self._module = []

    @property
    def next_word(self) -> str:
        """
        Get next token from buffer.
        Does not remove token.
        """
        tokens = self._buffer.split(' ')
        if len(tokens) == 0:
            return ''
        return tokens[0]

    @property
    def pop_next_word(self) -> str:
        """
        Get next token from buffer.
        Removes token.
        """
        tokens = self._buffer.split(' ')
        if len(tokens) == 0:
            return ''
        self._buffer = self._buffer[len(tokens[0]):]
        self._buffer = self._buffer.strip()
        return tokens[0]

    def evaluate(self) -> None:
        """
        Evaluate string stored in self._buffer.
        """
        # Ignore if empty
        if self._buffer == '':
            return
        # Clean the buffer
        self._buffer = self._buffer.strip()

        # Determine number of arguments each operator requires
        operators = {
            'BOSS': 1,
            'ADD': 2,
            'UPDATE': 2,
            'RESOLVE': 2,
            'DEMOTE': 1,
            'FIRE': 1,
            'VISUALIZE': 0,
        }
        # Determine if return None
        return_none = {
            'BOSS': True,
            'ADD': True,
            'UPDATE': True,
            'RESOLVE': False,
            'DEMOTE': True,
            'FIRE': True,
            'VISUALIZE': True
        }

        # Convert to Reverse Polish Notation
        tokens = tokenize(self._buffer)
        tokens.reverse()
        stack = []
        for token in tokens:
            if token in operators:
                new_token = ''
                for _ in range(0, operators[token]):
                    new_token += stack.pop() + ' '
                new_token += token
                stack.append(new_token)
            else:
                stack.append(token)
        self._buffer = stack.pop()

        # Evaluate
        tokens = tokenize(self._buffer)
        for token in tokens:
            if token in operators:
                # Make a backup
                operands = []
                for _ in range(0, operators[token]):
                    operands.append(stack.pop())
                operands.reverse()
                if operators[token] == 0:
                    if token == 'VISUALIZE':
                        self._company.visualize_graph()
                if operators[token] == 1:
                    expr = UnOp(self._company, token, Employee(self._company, operands[0]))
                    if return_none[token]:
                        expr.evaluate()
                    else:
                        stack.append(expr.evaluate())
                if operators[token] == 2:
                    expr = BinOp(self._company, Employee(self._company, operands[0]), token,
                                 Employee(self._company, operands[1]))
                    # Handle the only exception
                    if token == 'UPDATE':
                        expr = BinOp(self._company, Employee(self._company, operands[0]), token,
                                     Num(operands[1]))
                    if return_none[token]:
                        expr.evaluate()
                    else:
                        stack.append(expr.evaluate())
            else:
                stack.append(token)

    def start(self) -> None:
        """
        Start the REPL.
        """
        while True:
            line = input('BONE: ').split(' ')
            for i in range(0, len(line)):
                token = line[i]
                if token == 'QUIT':
                    return
                if token == 'END':
                    # This try except block is to allow
                    # graph.py to print the errors and for
                    # the REPL to not choke on an error
                    try:
                        self.evaluate()
                        for expr in self._module:
                            expr.evaluate()
                    except KeyError:
                        pass
                    finally:
                        self._buffer = ''
                        self._module = []
                    continue
                self._buffer += token + ' '


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod(verbose=True)

    python_ta.check_all(config={
        'max-line-length': 1000,
        # E9998 for IO, W0702 for try/except, E1136 R1710 for Optional[] typing,
        'disable': ['E9998', 'W0702', 'E1136', 'R1710'],
        'extra-imports': ['graph'],
        'max-nested-blocks': 6,
        'max-branches': 20
    })
