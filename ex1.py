class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

    def isEmpty(self):
        return self.front == self.rear

    def isFull(self):
        return (self.rear + 1) % self.max_size == self.front

    def enqueue(self, item):
        if not self.isFull():
            self.queue[self.rear] = item
            self.rear = (self.rear + 1) % self.max_size
        else:
            print("Queue is full")

    def dequeue(self):
        if not self.isEmpty():
            item = self.queue[self.front]
            self.front = (self.front + 1) % self.max_size
            return item
        else:
            print("Queue is empty")
            return None

def calc_height_diff(n):
    if n == None:
        return 0
    return calc_height_diff(n.left) - calc_height_diff(n.right)

def rotateLL(A):
    B = A.left
    A.left = B.right
    B.right = A
    return B

def rotateRR(A):
    B = A.right
    A.right = B.left
    B.left = A
    return B

def rotateRL(A):
    A.right = rotateLL(A.right)
    return rotateRR(A)

def rotateLR(A):
    A.left = rotateRR(A.left)
    return rotateLL(A)

def reBalance(root):
    hDiff = calc_height_diff(root)
    
    if hDiff > 1:
        if calc_height_diff(root.left) > 0:
            root = rotateLL(root)
        else:
            root = rotateLR(root)
    elif hDiff < -1:
        if calc_height_diff(root.right) < 0:
            root = rotateRR(root)
        else:
            root = rotateRL(root)
    
    return root

def insert_avl(root, node):
    if node.key<root.key:
        if root.left != None :
            root.left = insert_avl(root.left, node)
        else:
            root.left = node
        return reBalance(root)
    elif node.key > root.key:
        if root.right != None:
            root.right = insert_avl(root.right, node)
        else:
            root.right = node
        return reBalance(root);
    else:
        print("중복된 키 에러")
        
def levelorder(root):
    queue = CircularQueue(100)
    queue.enqueue(root)
    while not queue.isEmpty():
        n = queue.dequeue()
        if n is not None:
            print(n.key, end=' ')
            queue.enqueue(n.left)
            queue.enqueue(n.right)

if __name__ == "__main__":
    nodes = [7, 8, 9, 2, 1, 5, 3, 6, 4]
    
    root = None
    for i in nodes:
        n = BSTNode(i)
        root = insert_avl(root, n)
        print("AVL(%d):" % i, end='')
        levelorder(root)
        print()
    
    print("노드의 개수 =", count_node(root))
    print("단말의 개수 =", count_leaf(root))
    print("트리의 높이 =", count_height(root))
