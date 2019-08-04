/**!
 * @fileOverview Kickass library to create and place poppers near their reference elements.
 * @version 1.3.1
 * @license
 * Copyright (c) 2016 Federico Zivolo and contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import Popper from 'popper.js';

/**
 * Check if the given variable is a function
 * @method
 * @memberof Popper.Utils
 * @argument {Any} functionToCheck - variable to check
 * @returns {Boolean} answer to: is a function?
 */
function isFunction(functionToCheck) {
  const getType = {};
  return functionToCheck && getType.toString.call(functionToCheck) === '[object Function]';
}

var _extends = Object.assign || function (target) {
  for (var i = 1; i < arguments.length; i++) {
    var source = arguments[i];

    for (var key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        target[key] = source[key];
      }
    }
  }

  return target;
};

const DEFAULT_OPTIONS = {
  container: false,
  delay: 0,
  html: false,
  placement: 'top',
  title: '',
  template: '<div class="tooltip" role="tooltip"><div class="tooltip-arrow"></div><div class="tooltip-inner"></div></div>',
  trigger: 'hover focus',
  offset: 0,
  arrowSelector: '.tooltip-arrow, .tooltip__arrow',
  innerSelector: '.tooltip-inner, .tooltip__inner'
};

class Tooltip {
  /**
   * Create a new Tooltip.js instance
   * @class Tooltip
   * @param {HTMLElement} reference - The DOM node used as reference of the tooltip (it can be a jQuery element).
   * @param {Object} options
   * @param {String} options.placement='top'
   *      Placement of the popper accepted values: `top(-start, -end), right(-start, -end), bottom(-start, -end),
   *      left(-start, -end)`
   * @param {String} options.arrowSelector='.tooltip-arrow, .tooltip__arrow' - className used to locate the DOM arrow element in the tooltip.
   * @param {String} options.innerSelector='.tooltip-inner, .tooltip__inner' - className used to locate the DOM inner element in the tooltip.
   * @param {HTMLElement|String|false} options.container=false - Append the tooltip to a specific element.
   * @param {Number|Object} options.delay=0
   *      Delay showing and hiding the tooltip (ms) - does not apply to manual trigger type.
   *      If a number is supplied, delay is applied to both hide/show.
   *      Object structure is: `{ show: 500, hide: 100 }`
   * @param {Boolean} options.html=false - Insert HTML into the tooltip. If false, the content will inserted with `textContent`.
   * @param {String} [options.template='<div class="tooltip" role="tooltip"><div class="tooltip-arrow"></div><div class="tooltip-inner"></div></div>']
   *      Base HTML to used when creating the tooltip.
   *      The tooltip's `title` will be injected into the `.tooltip-inner` or `.tooltip__inner`.
   *      `.tooltip-arrow` or `.tooltip__arrow` will become the tooltip's arrow.
   *      The outermost wrapper element should have the `.tooltip` class.
   * @param {String|HTMLElement|TitleFunction} options.title='' - Default title value if `title` attribute isn't present.
   * @param {String} [options.trigger='hover focus']
   *      How tooltip is triggered - click, hover, focus, manual.
   *      You may pass multiple triggers; separate them with a space. `manual` cannot be combined with any other trigger.
   * @param {Boolean} options.closeOnClickOutside=false - Close a popper on click outside of the popper and reference element. This has effect only when options.trigger is 'click'.
   * @param {String|HTMLElement} options.boundariesElement
   *      The element used as boundaries for the tooltip. For more information refer to Popper.js'
   *      [boundariesElement docs](https://popper.js.org/popper-documentation.html)
   * @param {Number|String} options.offset=0 - Offset of the tooltip relative to its reference. For more information refer to Popper.js'
   *      [offset docs](https://popper.js.org/popper-documentation.html)
   * @param {Object} options.popperOptions={} - Popper options, will be passed directly to popper instance. For more information refer to Popper.js'
   *      [options docs](https://popper.js.org/popper-documentation.html)
   * @return {Object} instance - The generated tooltip instance
   */
  constructor(reference, options) {
    _initialiseProps.call(this);

    // apply user options over default ones
    options = _extends({}, DEFAULT_OPTIONS, options);

    reference.jquery && (reference = reference[0]);

    // cache reference and options
    this.reference = reference;
    this.options = options;

    // get events list
    const events = typeof options.trigger === 'string' ? options.trigger.split(' ').filter(trigger => ['click', 'hover', 'focus'].indexOf(trigger) !== -1) : [];

    // set initial state
    this._isOpen = false;
    this._popperOptions = {};

    // set event listeners
    this._setEventListeners(reference, events, options);
  }

  //
  // Public methods
  //

  /**
   * Reveals an element's tooltip. This is considered a "manual" triggering of the tooltip.
   * Tooltips with zero-length titles are never displayed.
   * @method Tooltip#show
   * @memberof Tooltip
   */


  /**
   * Hides an element’s tooltip. This is considered a “manual” triggering of the tooltip.
   * @method Tooltip#hide
   * @memberof Tooltip
   */


  /**
   * Hides and destroys an element’s tooltip.
   * @method Tooltip#dispose
   * @memberof Tooltip
   */


  /**
   * Toggles an element’s tooltip. This is considered a “manual” triggering of the tooltip.
   * @method Tooltip#toggle
   * @memberof Tooltip
   */


  /**
   * Updates the tooltip's title content
   * @method Tooltip#updateTitleContent
   * @memberof Tooltip
   * @param {String|HTMLElement} title - The new content to use for the title
   */


  //
  // Private methods
  //

  /**
   * Creates a new tooltip node
   * @memberof Tooltip
   * @private
   * @param {HTMLElement} reference
   * @param {String} template
   * @param {String|HTMLElement|TitleFunction} title
   * @param {Boolean} allowHtml
   * @return {HTMLElement} tooltipNode
   */
  _create(reference, template, title, allowHtml) {
    // create tooltip element
    const tooltipGenerator = window.document.createElement('div');
    tooltipGenerator.innerHTML = template.trim();
    const tooltipNode = tooltipGenerator.childNodes[0];

    // add unique ID to our tooltip (needed for accessibility reasons)
    tooltipNode.id = `tooltip_${Math.random().toString(36).substr(2, 10)}`;

    // set initial `aria-hidden` state to `false` (it's visible!)
    tooltipNode.setAttribute('aria-hidden', 'false');

    // add title to tooltip
    const titleNode = tooltipGenerator.querySelector(this.options.innerSelector);
    this._addTitleContent(reference, title, allowHtml, titleNode);

    // return the generated tooltip node
    return tooltipNode;
  }

  _addTitleContent(reference, title, allowHtml, titleNode) {
    if (title.nodeType === 1 || title.nodeType === 11) {
      // if title is a element node or document fragment, append it only if allowHtml is true
      allowHtml && titleNode.appendChild(title);
    } else if (isFunction(title)) {
      // if title is a function, call it and set textContent or innerHtml depending by `allowHtml` value
      const titleText = title.call(reference);
      allowHtml ? titleNode.innerHTML = titleText : titleNode.textContent = titleText;
    } else {
      // if it's just a simple text, set textContent or innerHtml depending by `allowHtml` value
      allowHtml ? titleNode.innerHTML = title : titleNode.textContent = title;
    }
  }

  _show(reference, options) {
    // don't show if it's already visible
    // or if it's not being showed
    if (this._isOpen && !this._isOpening) {
      return this;
    }
    this._isOpen = true;

    // if the tooltipNode already exists, just show it
    if (this._tooltipNode) {
      this._tooltipNode.style.visibility = 'visible';
      this._tooltipNode.setAttribute('aria-hidden', 'false');
      this.popperInstance.update();
      return this;
    }

    // get title
    const title = reference.getAttribute('title') || options.title;

    // don't show tooltip if no title is defined
    if (!title) {
      return this;
    }

    // create tooltip node
    const tooltipNode = this._create(reference, options.template, title, options.html);

    // Add `aria-describedby` to our reference element for accessibility reasons
    reference.setAttribute('aria-describedby', tooltipNode.id);

    // append tooltip to container
    const container = this._findContainer(options.container, reference);

    this._append(tooltipNode, container);

    this._popperOptions = _extends({}, options.popperOptions, {
      placement: options.placement
    });

    this._popperOptions.modifiers = _extends({}, this._popperOptions.modifiers, {
      arrow: {
        element: this.options.arrowSelector
      },
      offset: {
        offset: options.offset
      }
    });

    if (options.boundariesElement) {
      this._popperOptions.modifiers.preventOverflow = {
        boundariesElement: options.boundariesElement
      };
    }

    this.popperInstance = new Popper(reference, tooltipNode, this._popperOptions);

    this._tooltipNode = tooltipNode;

    return this;
  }

  _hide() /*reference, options*/{
    // don't hide if it's already hidden
    if (!this._isOpen) {
      return this;
    }

    this._isOpen = false;

    // hide tooltipNode
    this._tooltipNode.style.visibility = 'hidden';
    this._tooltipNode.setAttribute('aria-hidden', 'true');

    return this;
  }

  _dispose() {
    // remove event listeners first to prevent any unexpected behaviour
    this._events.forEach(({ func, event }) => {
      this.reference.removeEventListener(event, func);
    });
    this._events = [];

    if (this._tooltipNode) {
      this._hide();

      // destroy instance
      this.popperInstance.destroy();

      // destroy tooltipNode if removeOnDestroy is not set, as popperInstance.destroy() already removes the element
      if (!this.popperInstance.options.removeOnDestroy) {
        this._tooltipNode.parentNode.removeChild(this._tooltipNode);
        this._tooltipNode = null;
      }
    }
    return this;
  }

  _findContainer(container, reference) {
    // if container is a query, get the relative element
    if (typeof container === 'string') {
      container = window.document.querySelector(container);
    } else if (container === false) {
      // if container is `false`, set it to reference parent
      container = reference.parentNode;
    }
    return container;
  }

  /**
   * Append tooltip to container
   * @memberof Tooltip
   * @private
   * @param {HTMLElement} tooltipNode
   * @param {HTMLElement|String|false} container
   */
  _append(tooltipNode, container) {
    container.appendChild(tooltipNode);
  }

  _setEventListeners(reference, events, options) {
    const directEvents = [];
    const oppositeEvents = [];

    events.forEach(event => {
      switch (event) {
        case 'hover':
          directEvents.push('mouseenter');
          oppositeEvents.push('mouseleave');
          break;
        case 'focus':
          directEvents.push('focus');
          oppositeEvents.push('blur');
          break;
        case 'click':
          directEvents.push('click');
          oppositeEvents.push('click');
          break;
      }
    });

    // schedule show tooltip
    directEvents.forEach(event => {
      const func = evt => {
        if (this._isOpening === true) {
          return;
        }
        evt.usedByTooltip = true;
        this._scheduleShow(reference, options.delay, options, evt);
      };
      this._events.push({ event, func });
      reference.addEventListener(event, func);
    });

    // schedule hide tooltip
    oppositeEvents.forEach(event => {
      const func = evt => {
        if (evt.usedByTooltip === true) {
          return;
        }
        this._scheduleHide(reference, options.delay, options, evt);
      };
      this._events.push({ event, func });
      reference.addEventListener(event, func);
      if (event === 'click' && options.closeOnClickOutside) {
        document.addEventListener('mousedown', e => {
          if (!this._isOpening) {
            return;
          }
          const popper = this.popperInstance.popper;
          if (reference.contains(e.target) || popper.contains(e.target)) {
            return;
          }
          func(e);
        }, true);
      }
    });
  }

  _scheduleShow(reference, delay, options /*, evt */) {
    this._isOpening = true;
    // defaults to 0
    const computedDelay = delay && delay.show || delay || 0;
    this._showTimeout = window.setTimeout(() => this._show(reference, options), computedDelay);
  }

  _scheduleHide(reference, delay, options, evt) {
    this._isOpening = false;
    // defaults to 0
    const computedDelay = delay && delay.hide || delay || 0;
    window.setTimeout(() => {
      window.clearTimeout(this._showTimeout);
      if (this._isOpen === false) {
        return;
      }
      if (!document.body.contains(this._tooltipNode)) {
        return;
      }

      // if we are hiding because of a mouseleave, we must check that the new
      // reference isn't the tooltip, because in this case we don't want to hide it
      if (evt.type === 'mouseleave') {
        const isSet = this._setTooltipNodeEvent(evt, reference, delay, options);

        // if we set the new event, don't hide the tooltip yet
        // the new event will take care to hide it if necessary
        if (isSet) {
          return;
        }
      }

      this._hide(reference, options);
    }, computedDelay);
  }

  _updateTitleContent(title) {
    if (typeof this._tooltipNode === 'undefined') {
      if (typeof this.options.title !== 'undefined') {
        this.options.title = title;
      }
      return;
    }
    const titleNode = this._tooltipNode.parentNode.querySelector(this.options.innerSelector);
    this._clearTitleContent(titleNode, this.options.html, this.reference.getAttribute('title') || this.options.title);
    this._addTitleContent(this.reference, title, this.options.html, titleNode);
    this.options.title = title;
    this.popperInstance.update();
  }

  _clearTitleContent(titleNode, allowHtml, lastTitle) {
    if (lastTitle.nodeType === 1 || lastTitle.nodeType === 11) {
      allowHtml && titleNode.removeChild(lastTitle);
    } else {
      allowHtml ? titleNode.innerHTML = '' : titleNode.textContent = '';
    }
  }

}

/**
 * Title function, its context is the Tooltip instance.
 * @memberof Tooltip
 * @callback TitleFunction
 * @return {String} placement - The desired title.
 */

var _initialiseProps = function () {
  this.show = () => this._show(this.reference, this.options);

  this.hide = () => this._hide();

  this.dispose = () => this._dispose();

  this.toggle = () => {
    if (this._isOpen) {
      return this.hide();
    } else {
      return this.show();
    }
  };

  this.updateTitleContent = title => this._updateTitleContent(title);

  this._events = [];

  this._setTooltipNodeEvent = (evt, reference, delay, options) => {
    const relatedreference = evt.relatedreference || evt.toElement || evt.relatedTarget;

    const callback = evt2 => {
      const relatedreference2 = evt2.relatedreference || evt2.toElement || evt2.relatedTarget;

      // Remove event listener after call
      this._tooltipNode.removeEventListener(evt.type, callback);

      // If the new reference is not the reference element
      if (!reference.contains(relatedreference2)) {
        // Schedule to hide tooltip
        this._scheduleHide(reference, options.delay, options, evt2);
      }
    };

    if (this._tooltipNode.contains(relatedreference)) {
      // listen to mouseleave on the tooltip element to be able to hide the tooltip
      this._tooltipNode.addEventListener(evt.type, callback);
      return true;
    }

    return false;
  };
};

export default Tooltip;
//# sourceMappingURL=tooltip.js.map
