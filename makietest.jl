using Makie

scene = Scene()
data = Node(rand(1000))
t = lift(c -> length(c)-999, data)
y = lift(a -> to_value(data)[max(1, a-999):max(a, 1000)], t)
#x = lift(b -> collect(max(b-999, 1):max(b, 1000)), t)
#x = 1:1000 # for static axis
scene = AbstractPlotting.timeseries(y)
display(scene)
for i in 1:500
frametime = @elapsed begin
    push!(data, rand(1000))
    AbstractPlotting.update!(scene) #comment out update calls for fixed axis
    AbstractPlotting.update_limits!(scene)
    end
    sleep(max(0.015-frametime, 0))
    println(frametime)
end

using Gtk

win = GtkWindow("asdfasd", 400, 200)

b = GtkButton("Click Me")
c = GtkScaleLeaf(true, -10:0.1:10)
push!(win,c)


function on_button_clicked(w)
  println("The button has been clicked")
end
signal_connect(on_button_clicked, b, "clicked")

showall(win)

using Makie
import AbstractPlotting.timeseries
signal = Node(0.0)
scene = Scene()

@recipe(TimeSeries, signal) do scene
    Theme(
        history = 1000;
        default_theme(scene, Lines)...
    )
end

scene = timeseries(signal)
display(scene)
# @async is optional, but helps to continue evaluating more code
@async while isopen(scene)
    # aquire data from e.g. a sensor:
@time begin
    data = reverse(collect(0:1000))
    # update the signal
    for i in 1:1000
    signal[] = pop!(data)
    # sleep/ wait for new data/ whatever...
    # It's important to yield here though, otherwise nothing will be rendered
    yield()
end
end
end
