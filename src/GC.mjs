function getStaticMethods(Class) {
  return Object.getOwnPropertyNames(Class).filter(prop => prop !== "constructor" && typeof Class[prop] === "function");
}

class GarbageCollector {
  static add(...addList) {
    addList.flat(Infinity).forEach(obj => {
      GarbageCollector.objects.add(obj)
    })
  }

  static pushException(...exceptionList) {
    exceptionList.flat(Infinity).forEach(obj => {
      const val = GarbageCollector.whitelist.get(obj) || 0
      GarbageCollector.whitelist.set(obj, val + 1)
    })
  }

  static popException(...exceptionList) {
    exceptionList.flat(Infinity).forEach(obj => {
      const val = GarbageCollector.whitelist.get(obj) || 0
      GarbageCollector.whitelist.set(obj, val - 1)
      if (GarbageCollector.whitelist.get(obj) <= 0) {
        GarbageCollector.whitelist.delete(obj)
      }
    })
  }

  static flush() {
    const flushed = [...GarbageCollector.objects].filter(
      obj => !GarbageCollector.whitelist.has(obj)
    )
    flushed.forEach(obj => {
      if (typeof obj.delete === 'function') {
        obj.delete()
      }
      GarbageCollector.objects.delete(obj)
    })
    return flushed.length
  }

  /**
   * Reference bookkeeping
   */
  static set(ref, name, newObj) {
    if (ref[name]) {
      GarbageCollector.popException(ref[name])
    }
    GarbageCollector.pushException(newObj)
    ref[name] = newObj
  }

  /**
   * Equip class to add constructor feedback
   * @param  {Set} classes Set of all the classes names
   * @param  {object} Class class to wrap
   * @returns {object} wrapped class
   */
  static initClass(classes, Class) {
    const NewClass = function (...args) {
      const instance = new Class(...args)
      GarbageCollector.add(instance)
      return instance
    }
    const arr = [Class, Class.prototype]
    for (let idx in arr) {
      let obj = arr[idx]
      getStaticMethods(obj).forEach(method => {
        const fun = obj[method]
        const wrapper = function (...args) {
          const rtn = fun.call(this, ...args)
          if (rtn && classes.has(rtn?.constructor?.name)) {
            GarbageCollector.add(rtn)
          }
          return rtn
        }
        Object.assign(wrapper, fun);
        if (fun.overloadTable) wrapper.overloadTable = fun.overloadTable;
        obj[method] = wrapper;
      })
    }

    getStaticMethods(Class).forEach(method => {
      NewClass[method] = Class[method];
    })
    NewClass.prototype = Class.prototype
    return NewClass
  }
}

// Add static members
GarbageCollector.objects = new Set();
GarbageCollector.whitelist = new Map(); // Reference count

export default GarbageCollector
export { getStaticMethods }