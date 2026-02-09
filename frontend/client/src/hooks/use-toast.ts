import { useState, useEffect } from "react"

const TOAST_LIMIT = 1
// Toast auto-dismiss delay (ms)

type ToasterToast = {
  id: string
  title?: string
  description?: string
  action?: React.ReactNode
  variant?: "default" | "destructive"
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

let count = 0

function genId() {
  count = (count + 1) % Number.MAX_SAFE_INTEGER
  return count.toString()
}

type ActionType = {
  type: "ADD_TOAST" | "UPDATE_TOAST" | "DISMISS_TOAST"
  toast?: ToasterToast
  toastId?: string
}

let memoryState: { toasts: ToasterToast[] } = { toasts: [] }
let listeners: Array<(state: typeof memoryState) => void> = []

function dispatch(action: ActionType) {
  if (action.type === "ADD_TOAST" && action.toast) {
    memoryState = {
      ...memoryState,
      toasts: [action.toast, ...memoryState.toasts].slice(0, TOAST_LIMIT)
    }
  } else if (action.type === "DISMISS_TOAST") {
    memoryState = {
      ...memoryState,
      toasts: memoryState.toasts.map(t =>
        t.id === action.toastId || !action.toastId ? { ...t, open: false } : t
      )
    }
  } else if (action.type === "UPDATE_TOAST" && action.toast) {
    memoryState = {
      ...memoryState,
      toasts: memoryState.toasts.map(t =>
        t.id === action.toast!.id ? { ...t, ...action.toast } : t
      )
    }
  }
  listeners.forEach((listener) => listener(memoryState))
}

function toast({ ...props }: Omit<ToasterToast, "id">) {
  const id = genId()
  const update = (props: ToasterToast) =>
    dispatch({
      type: "UPDATE_TOAST",
      toast: { ...props, id },
    })
  const dismiss = () => dispatch({ type: "DISMISS_TOAST", toastId: id })

  dispatch({
    type: "ADD_TOAST",
    toast: {
      ...props,
      id,
      open: true,
      onOpenChange: (open: boolean) => {
        if (!open) dismiss()
      },
    },
  })

  return {
    id: id,
    dismiss,
    update,
  }
}

function useToast() {
  const [state, setState] = useState<{ toasts: ToasterToast[] }>(memoryState)

  useEffect(() => {
    listeners.push(setState)
    return () => {
      const index = listeners.indexOf(setState)
      if (index > -1) {
        listeners.splice(index, 1)
      }
    }
  }, [state])

  return {
    ...state,
    toast,
    dismiss: (toastId?: string) => dispatch({ type: "DISMISS_TOAST", toastId }),
  }
}

export { useToast, toast }
