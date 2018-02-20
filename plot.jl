using Gadfly
using Colors
using DataFrames
using Query

Gadfly.push_theme(Theme(major_label_font="Droid Sans",
                        minor_label_font="Droid Sans",
                        major_label_font_size=18pt, minor_label_font_size=16pt,
                        line_width=0.8mm, key_label_font="Droid Sans",
                        lowlight_color=c->RGBA{Float32}(c.r, c.g, c.b, 0.2),
                        key_label_font_size=14pt, key_position=:right))

colors = [colorant"#e41a1c", colorant"#377eb8", colorant"#4daf4a",
          colorant"#984ea3", colorant"#ff7f00", colorant"#ffff33"]

function std_or_z(x)
    s = std(x)
    if isnan(s)
        s = 0.0
    end
    s
end

function get_sizes(filename::String)
    names = [:id, :epochs, :gen, :size]
    res = readtable(filename, separator=',', header=false, names=names)

    stats = by(res, [:epochs, :gen], df->DataFrame(
        msize=mean(df[:size]), ssize=std_or_z(df[:size])))
    stats[:low_size] = stats[:msize] - 0.5 * stats[:ssize]
    stats[:high_size] = stats[:msize] + 0.5 * stats[:ssize]

    res, stats
end

function plot_sizes(stats::DataFrame, title::String="Boston Housing - GRN sizes",
                    logfile::String="boston_sizes.pdf";
                    xmin=minimum(stats[:gen]),
                    xmax=maximum(stats[:gen]),
                    ymin=minimum(stats[:low_size]),
                    ymax=maximum(stats[:high_size]),
                    ylabel="N",
                    xlabel="Generation",
                    key_position=:right)
    plt = plot(stats, x=:gen, y=:msize, ymin=:low_size, ymax=:high_size,
               color=:epochs, Geom.line, Geom.ribbon,
               Scale.color_discrete_manual(colors...),
               Guide.ylabel(ylabel), Guide.xlabel(xlabel),
               Coord.cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
               Guide.title(title))
    draw(PDF(logfile, 8inch, 6inch), plt);
    plt
end

function get_error_res(filename::String, maxgen=0)
    # names = [:m, :time, :gen, :error, :accuracy, :mean, :std, :test]
    names = [:id, :epochs, :time, :gen, :fit, :test]
    error_res = readtable(filename, separator=',', header=false, names=names)
    # error_res[:fit] = log10.(error_res[:fit])
    error_res[:epochs] = string.(error_res[:epochs])
    bests = by(error_res, [:id, :epochs, :gen], df->DataFrame(
        error=minimum(df[:fit]), test_fit=minimum(df[:test])))

    error_stats = by(bests, [:epochs, :gen], df->DataFrame(
        merror=mean(df[:error]), serror=std_or_z(df[:error])))
    error_stats[:low_error] = error_stats[:merror] - 0.5 * error_stats[:serror]
    error_stats[:high_error] = error_stats[:merror] + 0.5 * error_stats[:serror]

    error_res, error_stats
end

function get_serror_res(filename::String, maxgen=0)
    # names = [:m, :time, :gen, :error, :accuracy, :mean, :std, :test]
    names = [:id, :epochs, :time, :gen, :epoch, :fit, :test]
    error_res = readtable(filename, separator=',', header=false, names=names)
    error_res[:epochs] = string.(error_res[:epochs])
    # error_res[:fit] = log10.(error_res[:fit])
    bests = by(error_res, [:id, :epochs, :gen], df->DataFrame(
        error=minimum(df[:fit]), test_fit=minimum(df[:test])))

    error_stats = by(bests, [:epochs, :gen], df->DataFrame(
        merror=mean(df[:error]), serror=std_or_z(df[:error])))
    error_stats[:low_error] = error_stats[:merror] - 0.5 * error_stats[:serror]
    error_stats[:high_error] = error_stats[:merror] + 0.5 * error_stats[:serror]

    error_res, error_stats
end

function get_learn_res(filename::String)
    all_res = readtable(filename, separator=',', header=false,
                        names = [:type, :id, :evo_epoch, :tgen, :time, :gen,
                                 :epoch, :loss, :test])

    not_grns = @from i in all_res begin
        @where i.type != "grn"
        @select i
        @collect DataFrame
    end;

    l_grns = @from i in all_res begin
        @where (i.type == "grn" && i.evo_epoch == "10" && i.tgen == 50)
        @select i
        @collect DataFrame
    end;

    res = vcat(not_grns, l_grns)

    stats = by(res, [:type, :evo_epoch, :tgen, :epoch], df->DataFrame(
        mloss=mean(df[:loss]), sloss=std(df[:loss]),
        mtest=mean(df[:test]), stest=std(df[:test])))
    stats[:low_loss] = stats[:mloss] - 0.5 * stats[:sloss]
    stats[:high_loss] = stats[:mloss] + 0.5 * stats[:sloss]
    stats[:low_test] = stats[:mtest] - 0.5 * stats[:stest]
    stats[:high_test] = stats[:mtest] + 0.5 * stats[:stest]

    res, stats
end

function get_final(filename::String)
    names = [:type, :id, :epoch, :tgen, :time, :gen, :mse, :test]
    final = readtable(filename, separator=',', header=false, names=names)
    final[:epoch] = string.(final[:epoch])
    final[:tgen][final[:tgen] .== 49] = 50
    final
end

function plot_final(final::DataFrame, title::String="Boston Housing - best GRNs",
                    filename::String="boston_final.pdf")
    grns = @from i in final begin
        @where i.type == "grn"
        @select i
        @collect DataFrame
    end;
    grns[:Generation] = string.(grns[:tgen])
    grns[:E] = "Train"
    grns[:yax] = grns[:mse]
    cgrns = deepcopy(grns)
    cgrns[:yax] = cgrns[:test]
    cgrns[:E] = "Test"

    full = vcat(grns, cgrns)

    plt = plot(full, xgroup=:epoch, ygroup=:E, x=:Generation, y=:yax, color=:epoch,
                     Scale.y_log10,
                     Guide.ylabel("Mean Squared Error"),
                     Guide.xlabel("Generation by number of epochs during evolution"),
                     Guide.title(title),
                     Guide.xlabel(nothing),
                     style(key_position=:none),
                     Geom.subplot_grid(Geom.boxplot),
                     Scale.color_discrete_manual(colors...));
    draw(PDF(filename, 18inch, 8inch), plt);
    plt
end

function plot_comparison(final::DataFrame, title::String="Boston Housing",
                    filename::String="boston_comparison.pdf", tgen=50)
    not_grns = @from i in final begin
        @where i.type != "grn"
        @select i
        @collect DataFrame
    end;

    l_grns = @from i in final begin
        @where (i.type == "grn" && i.epoch == "10" && i.tgen == tgen)
        @select i
        @collect DataFrame
    end;

    f = vcat(not_grns, l_grns);
    f[:const] = ""

    plt = plot(f, xgroup=:type, x=:const, y=:test,
               Scale.y_log10,
               Guide.xlabel("Model"),
               Guide.ylabel("Mean Squared Error"),
               # Geom.boxplot,
               # Scale.x_discrete(order=[4, 1, 2, 3]),
               Geom.subplot_grid(Geom.boxplot),
               Guide.title(title),
               style(key_position=:none, default_color=colorant"#000000"),
               Scale.color_discrete_manual(colors...));
    draw(PDF(filename, 8inch, 6inch), plt);
    plt
end

function plot_error(error_stats::DataFrame, title::String="Air Quality Evolution",
                    logfile::String="error.pdf";
                    xmin=minimum(error_stats[:gen]),
                    xmax=maximum(error_stats[:gen]),
                    ymin=log10(minimum(error_stats[:low_error])),
                    ymax=log10(maximum(error_stats[:high_error])),
                    ylabel="Mean Squared Error",
                    xlabel="Generation",
                    key_position=:right)
    zerr = @from i in error_stats begin
        @where ((i.epochs == "0") && (i.gen == 0))
        @select [i.merror, i.low_error, i.high_error]
        @collect
    end

    error_stats[:merror][error_stats[:gen] .== 0] = zerr[1][1].value
    error_stats[:low_error][error_stats[:gen] .== 0] = zerr[1][2].value
    error_stats[:high_error][error_stats[:gen] .== 0] = zerr[1][3].value

    plt = plot(error_stats, x=:gen, y=:merror, ymin=:low_error, ymax=:high_error,
               color=:epochs, Geom.line, Geom.ribbon,
               Scale.color_discrete_manual(colors...),
               Scale.y_log10,
               Guide.ylabel(ylabel), Guide.xlabel(xlabel),
               Coord.cartesian(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
               Guide.title(title))
    draw(PDF(logfile, 8inch, 6inch), plt);
    plt
end

function plot_learn(learn_stats::DataFrame, title::String="Energy Training",
                    logfile::String="learn.pdf")

    plt = plot(learn_stats, x=:epoch, y=:mloss, ymin=:low_loss, ymax=:high_loss,
               color=:type, Geom.line, Geom.ribbon,
               Scale.color_discrete_manual(colors...),
               Guide.ylabel("Loss"),
               Guide.title(title))
    draw(PDF(logfile, 8inch, 6inch), plt);
    plt
end


function plot_learn_distributions(learn_res::DataFrame,
                                  title::String="Boston learning distributions",
                                  logfile::String="boston_learn_dist.pdf")

    downsampled = @from i in res begin
        @where ((mod(i.epoch, 10) == 1) && (i.epoch <= 51))
        @select i
        @collect DataFrame
    end;

    plt = plot(downsampled, xgroup=:epoch, x=:type, y=:loss, color=:type,
               Scale.y_log10,
               Guide.ylabel("Loss [log10]"),
               Geom.subplot_grid(Geom.boxplot), Guide.xlabel("Epoch"),
               Guide.title(title), Scale.color_discrete_manual(colors...));

    draw(PDF(logfile, 12inch, 6inch), plt);
    plt
end
