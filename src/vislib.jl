module vis
ENV["WAYLAND_DISPLAY"]="wayland-1"
ENV["DISPLAY"]=":0"
using LinearAlgebra
using GLMakie
using Graphs
using GraphMakie
using NetworkLayout
using SparseArrays


const OFF = RGBAf(0.3, 0.3, 0.3, 0.5)
const ON  = RGBAf(1.0, 0.05, 0.05, 1.0)

function plot_network_evolution(A, nodes, evfunc, dims=2)
    N = length(nodes)
    I, J, _ = findnz(triu(A, 1))
    g = SimpleGraph(N)
    for k in eachindex(I)
        add_edge!(g, I[k], J[k])
    end
    println(typeof(g))
    points = Stress(; dim=dims)(g)
    if dims == 3
        segments = Vector{Point3f}(undef, 2 * ne(g))
    else
        segments = Vector{Point2f}(undef, 2 * ne(g))
    end
    k = 1
    for e in edges(g)
        i = src(e)
        j = dst(e)
        segments[2k-1] = points[i]
        segments[2k]   = points[j]
        k += 1
    end

    colors = fill(OFF, N)
    color_obs = Observable(colors)

    fig = Figure()
    if dims == 3
        ax = Axis3(
            fig[1, 1],
            aspect = :data,
            perspectiveness = 0.5
        )
    else
        ax = Axis(
            fig[1, 1],
            aspect = DataAspect(),
            xgridvisible = false,
            ygridvisible = false,
            xticklabelsvisible = false,
            yticklabelsvisible = false
        )
    end
    hidedecorations!(ax)
    linesegments!(
        ax,
        segments;
        color = RGBAf(0.5, 0.5, 0.5, 0.1),
        linewidth = 0.5,
        transparency=true
    )
    scatter!(
        ax,
        points;
        color = color_obs,
        markersize = 10,
        transparency = true
    )
    display(fig)

    # for frame in 1:1000
    while true
        evfunc(nodes)
        @inbounds for i in 1:N
            # node_colors[i] = nodes[i] ? ON : OFF
            if nodes[i] == 1
                colors[i] = ON
            else
                colors[i] = OFF
            end
        end
        notify(color_obs)
        yield()
        sleep(0.01)
    end
end

function avalanche_plot(N, x, y, tau, D)
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xscale = log10,
        yscale = log10,
        xlabel = L"\text{Avalanche Size} / N^D",
        ylabel = L"\text{Probability} \times s^{\tau}",
    )


    τ = Observable(tau)
    D = Observable(D)

    sg = SliderGrid(fig[2, 1],
                    (label = "τ", range = 0:0.001:5, startvalue = τ),
                    (label = "D", range = -2.5:0.001:2.5, startvalue = D),
                    )

    on(sg.sliders[1].value) do val
        τ[] = val
    end
    on(sg.sliders[2].value) do val
        D[] = val
    end

    for i in eachindex(x)
        x_i = @lift [ x[i][j] / (N[i] ^ $(D)) for j in eachindex(x[i]) ]
        y_i = @lift [ y[i][j] * (x[i][j] ^ $(τ)) for j in eachindex(y[i]) ]
        scatterlines!(
            ax,
            x_i,
            y_i,
            marker = :x,
            label  = "N=$(N[i])"
        )
    end

    axislegend(ax)
    display(fig)
    fig
end

end
