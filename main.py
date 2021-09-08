"""
File to run when launching program.
"""
import graph
import parsing

if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'max-line-length': 1000,
        'extra-imports': ['graph', 'parsing'],
        'max-nested-blocks': 6,
        'max-branches': 20
    })
    while True:
        print('Welcome to BONE. Enter 1 to start fresh, 2 to start with a template company.')
        while True:
            choice = input()
            if not choice.isdigit():
                print('Please enter either 1 or 2.')
                continue
            choice = int(choice)
            if choice not in [1, 2]:
                print('Please enter either 1 or 2.')
                continue
            g = graph.Tree()
            repl = parsing.REPL(g)
            if choice == 1:
                repl.start()
                break
            if choice == 2:
                # Add sample company
                g.add_root(graph.Vertex('A', 1, 100))
                g.add_vertex(graph.Vertex('B', 2, 80))
                g.add_vertex(graph.Vertex('C', 3, 80))
                g.add_vertex(graph.Vertex('D', 4, 60))
                g.add_vertex(graph.Vertex('E', 5, 70))
                g.add_vertex(graph.Vertex('F', 6, 30))
                g.add_vertex(graph.Vertex('G', 7, 50))

                g.add_edge(1, 2)
                g.add_edge(1, 3)
                g.add_edge(2, 4)
                g.add_edge(2, 5)
                g.add_edge(3, 6)
                g.add_edge(5, 7)

                repl.start()
                break
