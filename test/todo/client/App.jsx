import React, { useEffect, useState } from "react"
import TopazCubeClient from "topazcube"
import "./index.css"
import { GripHorizontal, Trash2 } from "lucide-react"

function App() {
  // Client instance
  let [client, setClient] = useState({})
  const [isConnected, setIsConnected] = useState(false)

  // Document state (list of todos)
  let [doc, setDoc] = useState({})
  // Sorted todos by order field
  const sortedTodos = doc.todos ? [...doc.todos].filter((todo) => !todo.deleted).sort((a, b) => a.order - b.order) : []

  // New todo input
  const [newTodo, setNewTodo] = useState("")

  // For drag and drop reordering
  const [draggedItem, setDraggedItem] = useState(null)
  const [movingUp, setMovingUp] = useState(false)
  const [dragOverIndex, setDragOverIndex] = useState(null)

  // Initialize client
  useEffect(() => {
    class TodoClient extends TopazCubeClient {
      constructor() {
        super({
          url: "ws://192.168.0.200:4799",
        })
      }

      onConnect() {
        setIsConnected(true)
        this.subscribe("todos")
      }

      onDisconnect() {
        setIsConnected(false)
      }

      onMessage(message) {
        console.log("orig onMessage", message)
      }

      onChange(name) {
        console.log('onChange ez', this.document)
        setDoc({ ...this.document })
      }
    }

    let nc = new TodoClient()
    setClient(nc)
    nc.connect()

    return () => {
      nc.destroy()
    }
  }, [])

  // Add new todo
  const addTodo = () => {
    if (newTodo) {
      doc.todos.push({
        title: newTodo,
        completed: false,
        order: doc.todos.length,
      })
      setNewTodo("")
    }
  }

  // Drag and drop handlers

  const handleDragStart = (e, todo, index) => {
    setDraggedItem(todo)
    e.dataTransfer.effectAllowed = "move"
    // Required for Firefox
    e.dataTransfer.setData("text/plain", index)
  }

  const handleDragOver = (e, overTodo, overIndex) => {
    e.preventDefault()
    if (!draggedItem || draggedItem === overTodo) return
    setDragOverIndex(overIndex)
    if (draggedItem.order <= overTodo.order) {
      setMovingUp(false)
    } else {
      setMovingUp(true)
    }
  }

  const handleDragEnd = () => {
    setDraggedItem(null)
    setDragOverIndex(null)
  }

  const handleDrop = (e, dropTodo) => {
    e.preventDefault()
    if (!draggedItem || draggedItem === dropTodo) return

    // Get current orders
    const currentOrders = sortedTodos.map((todo) => todo.order)

    // Find the current order of dragged and drop items
    const draggedOrder = draggedItem.order
    const dropOrder = dropTodo.order

    // Reorder the items
    if (draggedOrder < dropOrder) {
      // Moving down
      doc.todos.forEach((todo) => {
        if (todo === draggedItem) {
          todo.order = dropOrder
        } else if (todo.order > draggedOrder && todo.order <= dropOrder) {
          todo.order--
        }
      })
    } else {
      // Moving up
      doc.todos.forEach((todo) => {
        if (todo === draggedItem) {
          todo.order = dropOrder
        } else if (todo.order >= dropOrder && todo.order < draggedOrder) {
          todo.order++
        }
      })
    }

    setDoc({ ...doc })
    setDraggedItem(null)
    setDragOverIndex(null)
  }

  return (
    <>
      <div className="mx-auto w-full md:max-w-2xl p-2">
        <h1>TopazCube Client Test</h1>
        <h2>Todo list: '{doc.title}'</h2>
        {sortedTodos.length > 0 ? (
          <div>
            <ul>
              {sortedTodos.map(
                (todo, index) =>
                  !todo.deleted && (
                    <li
                      key={index}
                      className={`flex items-center gap-2 py-1
                        ${index % 2 == 0 ? "bg-gray-200" : ""} ${
                        dragOverIndex === index ? "border-" + (movingUp ? "t" : "b") + "-3 border-blue-500" : ""
                      }`}
                      draggable={true}
                      onDragStart={(e) => handleDragStart(e, todo)}
                      onDragOver={(e) => handleDragOver(e, todo, index)}
                      onDragEnd={handleDragEnd}
                      onDrop={(e) => handleDrop(e, todo)}>
                      <button className="!cursor-move">
                        <GripHorizontal />
                      </button>
                      <input
                        type="checkbox"
                        className="cursor-pointer w-5 h-5"
                        checked={todo.completed}
                        onChange={() => {
                          todo.completed = !todo.completed
                        }}
                      />
                      <span className="flex-grow ml-2 cursor-move">
                        <span className={`${todo.completed ? "line-through opacity-30" : "font-bold"} cursor-text`}>
                          {todo.title}
                        </span>
                      </span>
                      <button
                        onClick={() => {
                          todo.deleted = true
                        }}>
                        <Trash2 />
                      </button>
                    </li>
                  )
              )}
            </ul>
          </div>
        ) : (
          <p>No todos yet. Add some!</p>
        )}
        <br />
        <div className="flex gap-2">
          <input
            type="text"
            className="flex-grow border-2 border-gray-400 px-2 py-1 rounded-md"
            placeholder="New Todo"
            value={newTodo}
            onChange={(e) => setNewTodo(e.target.value)}
          />
          <button
            className={`bg-gray-300 ${
              newTodo ? "text-black" : "text-gray-400 opacity-50 !cursor-not-allowed"
            } font-bold border-2 border-gray-400 px-4 py-2 rounded-md`}
            onClick={addTodo}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                addTodo()
              }
            }}>
            Add Todo
          </button>
        </div>
        <br />
        <pre>{JSON.stringify(doc, null, 2)}</pre>
        <br />
      </div>
      {!isConnected ? (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <h2 className="text-white text-xl font-bold">Connecting...</h2>
        </div>
      ) : (
        <></>
      )}
    </>
  )
}

export default App
